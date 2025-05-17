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
import os
import uuid
from collections import Counter
import requests
from serpapi.google_search import GoogleSearch
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.output_parsers import StrOutputParser
import json
import traceback

# Import Huawei OBS
from obs import ObsClient

# Import Ollama integration
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Import search tools
from langchain_community.tools import TavilySearchResults

from langgraph.graph import START, END, StateGraph

class Decision(BaseModel):
    classification: Literal["REAL", "FAKE"]
    confidence: int
    explanation: str
    sources: List[str]


# Define the state which will be passed around
class ImageClassificationState(TypedDict):
    image: str  # Now expects a local file path
    image_data: Dict[str, Any]
    obs_url: str  # URL of the image uploaded to Huawei OBS
    description: str
    web_scrape_results: Optional[Dict[str, Any]]
    search_query: str
    classification: Literal["Real", "Fake"]
    sources: Annotated[List[str], operator.add]
    decision: Optional[Dict[str, Any]]
    similar_images_count: int  # Number of visually similar images
    visual_match_urls: Annotated[List[str], operator.add]  # Top 10 visual matches' URLs
    visual_match_contents: Annotated[List[str], operator.add]  # Top 10 visual matches' website contents
    ai_generated_likelihood: float  # Likelihood of AI generation from SightEngine
    deepfake_likelihood: float  # Likelihood of deepfake from SightEngine
    is_portrait: bool  # Whether the image is a portrait

def load_image(state):
    # Get the local file path
    image_path = state["image"]
    
    try:
        # Load the image from the local path
        image = Image.open(image_path)
        
        # Convert to base64 for internal use
        buffered = BytesIO()
        image.save(buffered, format=image.format or "JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Configure OBS client
        ak = os.getenv("AccessKeyID")
        sk = os.getenv("SecretAccessKey")
        server = "https://obs.ap-southeast-3.myhuaweicloud.com"
        obsClient = ObsClient(access_key_id=ak, secret_access_key=sk, server=server)
        
        # Create a unique objectKey using UUID and original filename
        file_name = os.path.basename(image_path)
        unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID for brevity
        object_key = f"images/{unique_id}_{file_name}"
        
        # Upload to Huawei OBS
        bucket_name = "groupd"
        resp = obsClient.putFile(bucket_name, object_key, image_path)
        
        if resp.status < 300:
            # Construct public URL
            endpoint = "obs.ap-southeast-3.myhuaweicloud.com"
            public_url = f"https://{bucket_name}.{endpoint}/{object_key}"
            print(f"Upload succeeded: {public_url}")
        else:
            raise Exception(f"Upload failed with status {resp.status}")
        
        return {
            "image_data": {
                "success": True,
                "image_data": img_str,
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "path": image_path
            },
            "obs_url": public_url  # Store the public URL
        }
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        # Return error state that won't break the pipeline
        return {
            "image_data": {
                "success": False,
                "error": str(e),
                "image_data": "",  # Empty string instead of None
                "width": 0,
                "height": 0,
                "format": None,
                "path": image_path
            },
            "obs_url": ""  # Empty URL on failure
        }

def detect_ai_generation(state):
    try:
        # Get the local file path
        image_path = state["image"]
        
        # SightEngine API configuration
        params = {
            'models': 'genai',
            'api_user': '1614642259',  # Add your API user
            'api_secret': 'XkWh5ceXBK9dL9bR6YRemnwCsUv6oTRw'  # Add your API secret
        }
        
        # Open and send the file
        files = {'media': open(image_path, 'rb')}
        response = requests.post('https://api.sightengine.com/1.0/check.json', 
                                files=files, 
                                data=params)
        
        # Parse the response
        result = json.loads(response.text)
        
        # Check if the API call was successful
        if result.get('status') == 'success':
            ai_likelihood = result.get('type', {}).get('ai_generated', 0.0)
        else:
            ai_likelihood = 0.0
            print(f"SightEngine API error: {result}")
        
        return {
            "ai_generated_likelihood": ai_likelihood
        }
        
    except Exception as e:
        print(f"Error in AI generation detection: {str(e)}")
        return {
            "ai_generated_likelihood": 0.0  # Default to 0 on error
        }
    
# Define the deepfake detection tool
@tool
def check_deepfake(url: str) -> Dict[str, Any]:
    """
    Checks if a portrait image contains a deepfake using Sightengine API.
    Only use this for close-up portraits of a single person's face.
    
    Args:
        url: URL of the image to check
        
    Returns:
        Dictionary with deepfake detection results
    """
    try:
        params = {
            'url': url,
            'models': 'deepfake',
            'api_user': '1614642259',  # Add your API user
            'api_secret': 'XkWh5ceXBK9dL9bR6YRemnwCsUv6oTRw'  # Add your API secret
        }
        
        response = requests.get('https://api.sightengine.com/1.0/check.json', params=params)
        result = json.loads(response.text)
        
        if result.get('status') == 'success':
            deepfake_likelihood = result.get('type', {}).get('deepfake', 0.0)
            return {
                "success": True,
                "deepfake_likelihood": deepfake_likelihood,
                "raw_response": result
            }
        else:
            error_msg = result.get('error', {}).get('message', 'Unknown error')
            return {
                "success": False,
                "error": f"API error: {error_msg}",
                "deepfake_likelihood": 0.0
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "deepfake_likelihood": 0.0
        }    
    
def describe_image(state):
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
    description = response.content
    
    # Now determine if it's a portrait
    portrait_messages = [
        SystemMessage(content="""You are an AI assistant that determines if an image is a portrait.
        A portrait is defined as an image primarily featuring a single person's face or upper body.
        You should answer with just 'YES' if it's a portrait, or 'NO' if it's not."""),
        HumanMessage(content=f"Based on this description, is the image a portrait of a single person? Description: {description}")
    ]
    
    portrait_response = llm.invoke(portrait_messages)
    is_portrait = "YES" in portrait_response.content.upper()
    
    return {
        "description": description,
        "is_portrait": is_portrait
    }

def detect_deepfake(state):
    image_url = state.get("obs_url", "")
    
    try:
        deepfake_result = check_deepfake(image_url)
        deepfake_likelihood = deepfake_result.get("deepfake_likelihood", 0.0)
    except Exception as e:
        print(f"Error in deepfake detection: {str(e)}")
        deepfake_likelihood = 0.0
    
    return {
        "deepfake_likelihood": deepfake_likelihood
    }

def should_detect_deepfake(state):
    """Determines if deepfake detection should be performed based on whether the image is a portrait."""
    return state.get("is_portrait", False)

        
def load_webpage_content(url: str) -> str:
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        content = "\n\n".join([doc.page_content for doc in documents])
        return content[:10000]  # Limit content size
    except Exception as e:
        return f"Error loading webpage: {e}"

def reverse_image_search(state):
    # Use the OBS URL for reverse image search instead of local path
    image_url = state["obs_url"]
    
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
    ai_generated_likelihood = state.get("ai_generated_likelihood", 0.0)
    is_portrait = state.get("is_portrait", False)
    deepfake_likelihood = state.get("deepfake_likelihood", 0.0)

    # Prepare deepfake information for the prompt
    deepfake_info = ""
    if is_portrait:
        deepfake_info = f"""
        - Deepfake detection: Image was identified as a portrait
        - Deepfake likelihood: {deepfake_likelihood * 100:.2f}%"""
    
    prompt = f"""
    You are an image verification specialist. Your task is to determine whether the provided image is part of a real-world event or a fake/edited scene.

    IMAGE DESCRIPTION:
    {state["description"]}
    
    REVERSE IMAGE SEARCH RESULTS:
    - Number of visually similar images found: {similar_images_count}
    - Visual match URLs: {', '.join(state.get('visual_match_urls', []))}
    - Visual match contents:
    {visual_match_contents}
    
    IMAGE ANALYSIS:
    - AI generation likelihood: {ai_generated_likelihood * 100:.2f}%{deepfake_info}
    
    CRITICAL RULES:
    - Reverse image search is the MOST IMPORTANT signal.
    - If the number of visually similar images is less than 6 this could indicate fake news.
    - If the contents from the visual matches don't contain content similar to the image description, this is also likely to indicate fake news.
    - If contents mention "photoshopped", "not real", or similar phrases, that is strong evidence of fakery.
    - If AI generation likelihood is above 90%, consider this as strong evidence of potential fakery.
    - If the image is a portrait and deepfake likelihood is above 80%, consider this strong evidence of fakery.

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
    workflow.add_node("detect_deepfake", detect_deepfake)
    workflow.add_node("reverse_image_search", reverse_image_search)
    workflow.add_node("ai_detect", detect_ai_generation)
    workflow.add_node("classify_image", classify_image)
    
    # Define the edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "load_image")    
    workflow.add_edge("load_image", "describe_image")
    
    # Always run reverse_image_search and ai_detect
    workflow.add_edge("describe_image", "reverse_image_search")
    workflow.add_edge("load_image", "ai_detect")
    
    # Conditional edge for deepfake detection
    workflow.add_conditional_edges(
        "describe_image",
        should_detect_deepfake,
        {
            True: "detect_deepfake",
            False: "classify_image"
        }
    )
    
    # Connect from detect_deepfake to classify_image
    workflow.add_edge("detect_deepfake", "classify_image")
    
    # Connect remaining parallel branches to classify_image
    workflow.add_edge(["ai_detect", "reverse_image_search"], "classify_image")
    
    workflow.add_edge("classify_image", END)
    
    # Compile the graph
    return workflow.compile()

graph = create_image_classification_graph()