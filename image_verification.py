from typing import Dict, Any, Optional, List, Literal, Annotated
from typing_extensions import TypedDict
from langchain_community.document_loaders import WebBaseLoader
from pydantic import BaseModel, Field
import operator
import base64
from io import BytesIO
from PIL import Image
import re
import torch
import os

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.output_parsers import StrOutputParser

# Import Ollama integration
from langchain_ollama import OllamaLLM

# Import search tools
from langchain_community.tools import TavilySearchResults

from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

# Import the custom modules
from image_processor import preprocess_image
from bnext_classifier import BNextClassifier

# Initialize BNext classifier
classifier = BNextClassifier(model_path="models/bnext_l_pretrained.pth.tar")

class Decision(BaseModel):
    classification: Literal["REAL", "FAKE"]
    confidence: int
    explanation: str
    sources: List[str]


# Define the state which will be passed around
class ImageClassificationState(TypedDict):
    image: str
    image_features: Dict[str, Any]
    image_classes: List[Dict[str, Any]]
    description: str
    web_scrape_results: Optional[Dict[str, Any]]
    search_query: str
    classification: Literal["Real", "Fake"]
    sources: Annotated[List[str], operator.add]
    decision: Optional[Dict[str, Any]]

def extract_image_features(state):
    """Extract features from the image using BNext-L for deepfake detection"""
    image_path = state["image"]
    
    # Preprocess image
    image_tensor = preprocess_image(image_path)
    
    # Get binary classification (real/fake)
    classification_result = classifier.classify(image_tensor)
    
    # Get image metadata
    img = Image.open(image_path)
    
    return {
        "image_features": {
            "success": True,
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "is_fake": classification_result['is_fake'],
            "confidence": classification_result['confidence']
        },
        "image_classes": classification_result
    }

def create_image_description(state):
    """Create description from image features for a potentially deepfake image"""
    
    # Initialize Ollama with llama3 model
    llm = OllamaLLM(model="llama3")
    
    # Extract classification result
    classification = state["image_classes"]
    classification_info = f"- Classification: {classification['class_name']}\n- Confidence: {classification['confidence']:.2f}\n"
    
    if classification['full_probs']:
        prob_info = f"- Probability Real: {classification['full_probs']['real']:.2f}\n- Probability Fake: {classification['full_probs']['fake']:.2f}"
        classification_info += prob_info
    
    # Get image dimensions
    features = state["image_features"]
    dimensions = f"Image dimensions: {features['width']} x {features['height']} ({features['format']})"
    
    # Create prompt template for generating description without classification
    template = f"""You are an AI assistant that provides detailed and objective descriptions of images.
    Focus on describing visual elements that can be verified:
    - People or notable figures in the image and their names
    - Location and setting 
    - Any visible text or signs
    - Events or activities depicted
    - Distinctive objects or landmarks
    - Approximate time period or date indicators
    
    This image may potentially be a deepfake or doctored image. Your model analysis detected:
    {classification_info}
    {dimensions}
    
    Provide a detailed VISUAL description of what appears in this image, without speculating whether it's real or fake.
    Focus only on what can be visually observed, not on authenticity assessment:"""
    
    description = llm.invoke(template)
    
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
    
    # Get BNext-L binary classification result
    bnext_classification = state["image_classes"]
    bnext_features = state["image_features"]
    model_assessment = f"""
    BNext-L Classification: {bnext_classification['class_name']}
    Model Confidence: {bnext_classification['confidence']:.2f}
    Probability Real: {bnext_classification['full_probs']['real']:.2f}
    Probability Fake: {bnext_classification['full_probs']['fake']:.2f}
    """
    
    sources_text = "\n".join([f"- {source}" for source in state["sources"]])
    
    # Add web content if available
    web_content_text = ""
    if state.get("web_scrape_results") and state["web_scrape_results"].get("contents"):
        web_content_text = "\n\n".join(state["web_scrape_results"]["contents"])
    
    prompt = f"""You are an image verification specialist. Determine if this image represents a real event or is fake/doctored.

        IMAGE DESCRIPTION:
        {state["description"]}
        
        DEEPFAKE DETECTION MODEL ASSESSMENT:
        {model_assessment}

        WEB SOURCES:
        {sources_text}
        
        WEB CONTENT:
        {web_content_text}

        Analyze whether the sources confirm or refute the authenticity of what's described in the image.
        Consider:
        - If credible sources mention the event/scene described
        - Consistency between image description and information from reliable sources
        - Evidence of manipulation or misrepresentation
        - Presence in fact-checking websites
        - The BNext-L model assessment (this is an AI model specifically trained for deepfake detection)

        Weigh the evidence from both the deepfake detection model and web sources.
        
        Based only on this information, provide your verdict in this exact format:
        CLASSIFICATION: [REAL or FAKE]
        CONFIDENCE: [0-100]
        EXPLANATION: [Your detailed explanation with references to specific sources and the model assessment]"""

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
    
    # Combine model assessment with LLM analysis
    # If BNext is very confident and LLM couldn't find evidence, lean toward model assessment
    if bnext_classification['confidence'] > 0.85 and confidence < 60:
        # Adjust confidence based on model confidence
        adjusted_confidence = int(confidence * 0.3 + bnext_classification['confidence'] * 100 * 0.7)
        confidence = min(100, adjusted_confidence)
        
        # If model and LLM disagree, prefer model but note disagreement
        if (bnext_classification['class_name'] == "FAKE" and classification == "REAL") or \
           (bnext_classification['class_name'] == "REAL" and classification == "FAKE"):
            classification = bnext_classification['class_name']
            explanation = f"NOTE: The BNext deepfake detection model strongly indicates this image is {bnext_classification['class_name'].lower()} " \
                          f"with high confidence ({bnext_classification['confidence']:.2f}), which overrides web evidence.\n\n" + explanation
    
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
    workflow.add_node("extract_image_features", extract_image_features)
    workflow.add_node("create_image_description", create_image_description)
    workflow.add_node("optimize_search_query", optimize_search_query)
    workflow.add_node("webscrape_content", webscrape_content)
    workflow.add_node("classify_image", classify_image)
    
    # Define the edges between nodes (sequential flow)
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "extract_image_features")
    workflow.add_edge("extract_image_features", "create_image_description")
    workflow.add_edge("create_image_description", "optimize_search_query")
    workflow.add_edge("optimize_search_query", "webscrape_content")
    workflow.add_edge("webscrape_content", "classify_image")
    workflow.add_edge("classify_image", END)
    
    # Compile the graph
    return workflow.compile()

# Create the graph, to launch using 'langgraph dev'
graph = create_image_classification_graph()