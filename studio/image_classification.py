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

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.output_parsers import StrOutputParser

# Import Ollama integration
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage

# Import search tools
from langchain_community.tools import TavilySearchResults
from langchain_google_community import GoogleSearchAPIWrapper

from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

class Decision(BaseModel):
    classification: Literal["REAL", "FAKE"]
    confidence: int
    explanation: str
    sources: List[str]


# Define the state which will be passed around
class ImageClassificationState(TypedDict):
    image: str
    image_data: Dict[str, Any]
    description: str
    web_scrape_results: Optional[Dict[str, Any]]
    search_query: str
    classification: Literal["Real", "Fake"]
    sources: Annotated[List[str], operator.add]
    decision: Optional[Dict[str, Any]]

def load_image(state):
    
    image = Image.open(state["image"])	
    buffered = BytesIO()
    image.save(buffered, format=image.format or "JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
    image_data={
            "success": True,
            "image_data": img_str,
            "width": image.width,
            "height": image.height,
            "format": image.format
        }
    
    return {"image_data": image_data}
        


def describe_image(state):
    
    # Initialize Ollama with llava model
    llm = OllamaLLM(model="llava")
    
    # Get the base64 image from state
    image_base64 = state["image_data"]["image_data"]
    
    # Create prompt template for image description
    template = """You are an AI assistant that provides detailed descriptions of images.
    Focus on key elements that can be verified:
    - People or notable figures in the image and their names
    - Location and setting
    - Any visible text or signs
    - Events or activities depicted
    - Distinctive objects or landmarks
    - Approximate time period or date indicators
    
    Please provide a detailed factual description of this image:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Bind the image to the LLM
    llm_with_image = llm.bind(images=[image_base64])
    
    # Create the chain
    chain = prompt | llm_with_image
    description = chain.invoke({})
    
    return {"description": description}

def optimize_search_query(state):
    
    llm = OllamaLLM(model="llama3")
    
    prompt = f"""Given this image description, create a concise search query that will be used to verify if the image represents a real event.
    The query should focus on the most distinctive and verifiable elements, include names, locations, or dates if present, and be under 10 words.
    Return only the search query.
    
    IMAGE DESCRIPTION:
    {state["description"]}
    
    SEARCH QUERY:"""
    
    search_query = llm.invoke(prompt).strip()
    
    # Clean up the query (remove any additional text the model might add)
    if len(search_query.split()) > 30:
        search_query = " ".join(search_query.split()[:30])
    
    return {"search_query": search_query}
        
# Function to perform web scraping using Tavily
def load_webpage_content(url: str) -> str:
    """Load and extract content from a webpage URL."""
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        content = "\n\n".join([doc.page_content for doc in documents])
        return content[:10000]  # Limit content size
    except Exception as e:
        return f"Error loading webpage: {e}"

# Function to perform web scraping using Tavily
def webscrape_content(state):
    query = state['search_query']
    tavily_search = TavilySearchResults(max_results=5, include_raw_content=True)
    tavily_results = tavily_search.invoke({"query": query})
    
    sources = []
    web_contents = []
    
    for result in tavily_results:
        url = result.get("url")
        if url:
            sources.append(url)
            
            # Get content directly from Tavily result if available
            if result.get("content"):
                content = f"Source ({url}):\n{result.get('content')[:5000]}..."
                web_contents.append(content)

            else:
                try:
                    page_content = load_webpage_content(url)
                    content = f"Source ({url}):\n{page_content[:5000]}..."
                    web_contents.append(content)
                except Exception as e:
                    web_contents.append(f"Source ({url}):\nFailed to load content: {str(e)}")
    
    return {
        "sources": sources,
        "web_scrape_results": {
            "contents": web_contents
        }
    }

def classify_image(state):
    result_dict = {}
    
    llm = OllamaLLM(model="llama3")
    
    sources_text = "\n".join([f"- {source}" for source in state["sources"]])
    
    # Add web content if available
    web_content_text = ""
    if state.get("web_scrape_results") and state["web_scrape_results"].get("contents"):
        web_content_text = "\n\n".join(state["web_scrape_results"]["contents"])
    
    prompt = f"""You are an image verification specialist. Determine if this image represents a real event or fake news.

        IMAGE DESCRIPTION:
        {state["description"]}

        WEB SOURCES:
        {sources_text}
        
        WEB CONTENT:
        {web_content_text}

        Analyze if the sources confirm or refute the authenticity of what's described in the image.
        Consider:
        - If credible sources mention the event/scene described
        - Consistency between image description and information from reliable sources
        - Evidence of manipulation or misrepresentation
        - Presence in fact-checking websites

        Based only on this information, provide your verdict in this exact format:
        CLASSIFICATION: [REAL or FAKE]
        CONFIDENCE: [0-100]
        EXPLANATION: [Your detailed explanation with references to specific sources]"""

    analysis = llm.invoke(prompt)
    
    # Parse the analysis to extract classification, confidence, and explanation
    classification = "UNKNOWN"
    confidence = 0
    explanation = analysis
    
    # Extract classification
    classification_match = re.search(r'CLASSIFICATION:\s*(REAL|FAKE)', analysis, re.IGNORECASE)
    if classification_match:
        classification = classification_match.group(1).upper()
    
    # Extract confidence
    confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', analysis)
    if confidence_match:
        confidence = int(confidence_match.group(1))
        # Ensure confidence is within valid range
        confidence = max(0, min(100, confidence))
    
    # Extract explanation
    explanation_match = re.search(r'EXPLANATION:(.*)', analysis, re.DOTALL)
    if explanation_match:
        explanation = explanation_match.group(1).strip()
    
    decision = Decision(
        classification=classification,
        confidence=confidence,
        explanation=explanation,
        sources=state["sources"]
    )

    result_dict["decision"] = decision.model_dump()
    
    return result_dict


def create_image_classification_graph():
    workflow = StateGraph(ImageClassificationState)
    
    # Add nodes
    workflow.add_node("initialize", lambda state: state)
    workflow.add_node("load_image", load_image)
    workflow.add_node("describe_image", describe_image)
    workflow.add_node("optimize_search_query", optimize_search_query)
    workflow.add_node("webscrape_content", webscrape_content)
    workflow.add_node("classify_image", classify_image)
    
    # Define the edges between nodes (sequential flow)
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "load_image")
    workflow.add_edge("load_image", "describe_image")
    workflow.add_edge("describe_image", "optimize_search_query")
    workflow.add_edge("optimize_search_query", "webscrape_content")
    workflow.add_edge("webscrape_content", "classify_image")
    workflow.add_edge("classify_image", END)
    
    # Compile the graph
    return workflow.compile()

# Create the graph, to launch using 'langgraph dev'
graph = create_image_classification_graph()