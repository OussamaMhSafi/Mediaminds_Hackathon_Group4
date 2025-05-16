from typing import Dict, Any, Optional, List, Literal, Annotated
from typing_extensions import TypedDict
from langchain_community.document_loaders import WebBaseLoader
from pydantic import BaseModel, Field
from enum import Enum
import operator
import base64
from io import BytesIO
from PIL import Image
import re
from collections import Counter
import requests
from serpapi import GoogleSearch
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.output_parsers import StrOutputParser

# Import Ollama integration
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Import search tools
from langchain_community.tools import TavilySearchResults

from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

class Decision(BaseModel):
    classification: Literal["REAL", "FAKE"]
    confidence: int
    explanation: str
    sources: List[str]


# Define the state which will be passed around
class ImageClassificationState(TypedDict):
    image: str  # Now expects a URL to the image
    image_data: Dict[str, Any]
    description: str
    web_scrape_results: Optional[Dict[str, Any]]
    search_query: str
    classification: Literal["Real", "Fake"]
    sources: Annotated[List[str], operator.add]
    decision: Optional[Dict[str, Any]]
    similar_images_count: int  # Number of visually similar images
    visual_match_urls: Annotated[List[str], operator.add]  # Top 10 visual matches' URLs
    visual_match_contents: Annotated[List[str], operator.add]  # Top 10 visual matches' website contents

def load_image(state):
    
    # old image import using local image file
    """
    image = Image.open(state["image"])	
    buffered = BytesIO()
    image.save(buffered, format=image.format or "JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    """

    image_url = state["image"]
    
    try:
        # Download image from URL with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Verify content type is an image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise ValueError(f"URL does not point to an image. Content-Type: {content_type}")
        
        # Convert to Image
        image = Image.open(BytesIO(response.content))
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format=image.format or "JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            "image_data": {
                "success": True,
                "image_data": img_str,
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "url": image_url
            }
        }
    
    except Exception as e:
        print(f"Error loading image from URL: {str(e)}")  # Add debugging
        # Return error state that won't break the pipeline
        return {
            "image_data": {
                "success": False,
                "error": str(e),
                "image_data": "",  # Empty string instead of None
                "width": 0,
                "height": 0,
                "format": None,
                "url": image_url
            }
        }
    
def describe_image(state):
    # Use an OpenAI model that supports vision
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Get the base64 image from state
    image_base64 = state["image_data"]["image_data"]
    
    # Create messages with the image directly for OpenAI
    messages = [
        SystemMessage(content="You are an AI assistant that provides detailed descriptions of images."),
        HumanMessage(content=[
            {
                "type": "text",
                "text": """Focus on key elements that can be verified:
                - People or notable figures in the image and their names
                - Location and setting
                - Any visible text or signs
                - Events or activities depicted
                - Distinctive objects or landmarks
                - Approximate time period or date indicators
                
                Please provide a detailed factual description of this image:"""
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            }
        ])
    ]
    
    # Invoke the model directly with the messages
    response = llm.invoke(messages)
    
    return {"description": response.content}
        
def load_webpage_content(url: str) -> str:

    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        content = "\n\n".join([doc.page_content for doc in documents])
        return content[:10000]  # Limit content size
    except Exception as e:
        return f"Error loading webpage: {e}"

def reverse_image_search(state):

    image_url = state["image"]
    
    try:
        # Perform reverse image search using SerpAPI directly with the provided URL
        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": "c8c0a3f93eb0b6660ec0a33251fd4b61ecf4c6eec271c877997813fbca69a231"
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        visual_matches = results["visual_matches"]
        similar_images_count = len(visual_matches)
        
        # Get top 10 visual match URLs
        top_matches = visual_matches[:10] if similar_images_count >= 10 else visual_matches
        visual_match_urls = [match.get("link", "") for match in top_matches]
        
        # Get content from the top visual match URLs
        visual_match_contents = []
        for url in visual_match_urls:
            if url:
                try:
                    loader = WebBaseLoader(url)
                    documents = loader.load()
                    content = "\n\n".join([doc.page_content for doc in documents])
                    visual_match_contents.append(f"Source ({url}):\n{content[:5000]}...") 
                except Exception as e:
                    visual_match_contents.append(f"Source ({url}):\nFailed to load content: {str(e)}")
            else:
                visual_match_contents.append("No valid URL")
        
        return {
            "similar_images_count": similar_images_count,
            "visual_match_urls": visual_match_urls,
            "visual_match_contents": visual_match_contents
        }
    
    except Exception as e:
        # Return empty results with error information if the search fails
        return {
            "similar_images_count": 0,
            "visual_match_urls": [],
            "visual_match_contents": [f"Error performing reverse image search: {str(e)}"]
        }

def classify_image(state):
    result_dict = {}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    similar_images_count = state.get("similar_images_count", 0)
    visual_match_contents = "\n".join(state.get("visual_match_contents", []))

    prompt = f"""
    You are an image verification specialist. Your task is to determine whether the provided image is part of a real-world event or a fake/edited scene.

    IMAGE DESCRIPTION:
    {state["description"]}
    
    REVERSE IMAGE SEARCH RESULTS:
    - Number of visually similar images found: {similar_images_count}
    - Visual match URLs: {', '.join(state.get('visual_match_urls', []))}
    - Visual match contents:
    {visual_match_contents}

    CRITICAL RULES:
    - Reverse image search is the MOST IMPORTANT signal.
    - If the number of visually similar images is less than 10 always classify as FAKE.
    - If the contents from the visual matches don't contain content similar to the image description, this is also likely to indicate fake news
    - If contents mention "photoshopped", "not real", or similar phrases, that is strong evidence of fakery.

    Format:
    CLASSIFICATION: [REAL or FAKE]
    CONFIDENCE: [0-100]
    EXPLANATION: [Detailed justification with evidence]
    """

    analysis = llm.invoke(prompt)
    analysis_text = analysis.content

    # Parse results
    classification = "UNKNOWN"
    confidence = 0
    explanation = analysis_text

    classification_match = re.search(r'CLASSIFICATION:\s*(REAL|FAKE)', analysis_text, re.IGNORECASE)
    if classification_match:
        classification = classification_match.group(1).upper()

    confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', analysis_text)
    if confidence_match:
        confidence = int(confidence_match.group(1))
        confidence = max(0, min(100, confidence))

    explanation_match = re.search(r'EXPLANATION:(.*)', analysis_text, re.DOTALL)
    if explanation_match:
        explanation = explanation_match.group(1).strip()

    decision = Decision(
        classification=classification,
        confidence=confidence,
        explanation=explanation,
        sources=state["sources"] + state.get("visual_match_urls", [])
    )

    result_dict["decision"] = decision.model_dump()
    return result_dict

def create_image_classification_graph():
    workflow = StateGraph(ImageClassificationState)
    
    # Add nodes
    workflow.add_node("initialize", lambda state: state)
    workflow.add_node("load_image", load_image)
    workflow.add_node("describe_image", describe_image)
    workflow.add_node("reverse_image_search", reverse_image_search)
    workflow.add_node("classify_image", classify_image)
    
    # Define the edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "load_image")    
    workflow.add_edge("load_image", "describe_image")
    workflow.add_edge("describe_image", "reverse_image_search")
    workflow.add_edge("reverse_image_search", "classify_image")
    
    workflow.add_edge("classify_image", END)
    
    # Compile the graph
    return workflow.compile()

graph = create_image_classification_graph()