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
from obs import ObsClient
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools import TavilySearchResults
from langgraph.graph import START, END, StateGraph
from langchain.agents import AgentExecutor, create_openai_functions_agent

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
    agent_reasoning: Optional[str]  # Store agent's reasoning process


def load_image(state):

    image_path = state["image"]
    
    try:

        image = Image.open(image_path)
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
        unique_id = str(uuid.uuid4())[:8]
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
            "obs_url": public_url
        }
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        # Return error state that won't break the pipeline
        return {
            "image_data": {
                "success": False,
                "error": str(e),
                "image_data": "",
                "width": 0,
                "height": 0,
                "format": None,
                "path": image_path
            },
            "obs_url": "" 
        }


# Tool 1: Analyze image for content and portrait detection
@tool
def analyze_image_description(image_base64: str, image_url: str = None) -> Dict[str, Any]:
    """
    Analyzes an image to provide a detailed description and determines if it's a portrait.
    
    Args:
        image_base64: Base64-encoded image data
        image_url: Alternative URL of the image if base64 fails
        
    Returns:
        Dictionary with description and portrait detection result
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    if image_base64 and len(image_base64) > 100:  
        if "base64," in image_base64:
            image_base64 = image_base64.split("base64,")[1]
        
        # Create messages with the properly formatted base64 image
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
    elif image_url:
        # Fallback to using the image URL if base64 is unavailable
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
                        "url": image_url
                    }
                }
            ])
        ]
    else:
        return {
            "error": "No valid image data provided",
            "description": "Unable to analyze image: no valid image data provided",
            "is_portrait": False
        }
    
    try:
        # Invoke the model directly with the messages
        response = llm.invoke(messages)
        description = response.content
        
        # Now determine if it's a portrait
        portrait_messages = [
            SystemMessage(content="""You are an AI assistant that determines if an image is a portrait.
            A portrait is defined as an image primarily featuring a single person's face or upper body.
            Be flexible in your conclusion of what constitutes a portrait.
            You should answer with just 'YES' if it's a portrait, or 'NO' if it's not."""),
            HumanMessage(content=f"Based on this description, is the image a portrait of a single person? Description: {description}")
        ]
        
        portrait_response = llm.invoke(portrait_messages)
        is_portrait = "YES" in portrait_response.content.upper()
        
        return {
            "description": description,
            "is_portrait": is_portrait
        }
    except Exception as e:
        print(f"Error in image analysis: {str(e)}")
        return {
            "error": str(e),
            "description": "Error analyzing image",
            "is_portrait": False
        }


# Tool 2: Check if image was AI-generated
@tool
def check_ai_generation(image_path: str) -> Dict[str, Any]:
    """
    Checks if an image was likely generated by AI using SightEngine API.
    
    Args:
        image_path: Local file path to the image
        
    Returns:
        Dictionary with AI generation likelihood score
    """
    try:
        # SightEngine API configuration
        params = {
            'models': 'genai',
            'api_user': os.getenv("SightengineUserKey"),
            'api_secret': os.getenv("SightengineSecretKey") 
        }
        
        files = {'media': open(image_path, 'rb')}
        response = requests.post('https://api.sightengine.com/1.0/check.json', 
                               files=files, 
                               data=params)
        
        result = json.loads(response.text)
        
        if result.get('status') == 'success':
            ai_likelihood = result.get('type', {}).get('ai_generated', 0.0)
            return {
                "ai_generated_likelihood": ai_likelihood
            }
        else:
            return {
                "error": f"SightEngine API error: {result}",
                "ai_generated_likelihood": 0.0
            }
        
    except Exception as e:
        print(f"Error in AI generation detection: {str(e)}")
        return {
            "error": str(e),
            "ai_generated_likelihood": 0.0 
        }


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
            'api_user': os.getenv("SightengineUserKey"),
            'api_secret': os.getenv("SightengineSecretKey")
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


# Tool 4: Perform reverse image search
@tool
def perform_reverse_image_search(image_url: str) -> Dict[str, Any]:
    """
    Performs reverse image search using Google Lens via SerpAPI.
    
    Args:
        image_url: URL of the image to search
        
    Returns:
        Dictionary with search results including similar images and their contents
    """
    try:
        # Perform reverse image search using SerpAPI directly with the provided URL
        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": os.getenv("SerpAPIKey"),
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        visual_matches = results.get("visual_matches", [])
        similar_images_count = len(visual_matches)
        
        # Get top 10 visual match URLs
        top_matches = visual_matches[:5] if similar_images_count >= 5 else visual_matches
        visual_match_urls = [match.get("link", "") for match in top_matches]
        
        # Web scrape from the top visual match URLs
        visual_match_contents = []
        for url in visual_match_urls:
            if url:
                try:
                    loader = WebBaseLoader(
                        url,
                        requests_kwargs={"timeout": 10}
                    )
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
            "error": str(e),
            "similar_images_count": 0,
            "visual_match_urls": [],
            "visual_match_contents": [f"Error performing reverse image search: {str(e)}"]
        }


# Tool 5: Make final classification
@tool
def make_classification_decision(
    description: str, 
    similar_images_count: int,
    visual_match_urls: List[str],
    visual_match_contents: List[str],
    ai_generated_likelihood: float,
    is_portrait: bool,
    deepfake_likelihood: float = 0.0
) -> Dict[str, Any]:
    """
    Makes a final determination if an image is real or fake based on all evidence.
    
    Args:
        description: Description of the image
        similar_images_count: Number of similar images found
        visual_match_urls: URLs of similar images
        visual_match_contents: Contents of websites with similar images
        ai_generated_likelihood: Likelihood the image was AI-generated (0-1)
        is_portrait: Whether the image is a portrait
        deepfake_likelihood: Likelihood of deepfake if portrait (0-1)
        
    Returns:
        Dictionary with classification decision
    """

    result_dict = {}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    visual_match_contents_str = "\n".join(visual_match_contents[:3])  # Limit to first 3 for prompt size
    
    # Prepare deepfake information for the prompt
    deepfake_info = ""
    if is_portrait:
        deepfake_info = f"""
        - Deepfake detection: Image was identified as a portrait
        - Deepfake likelihood: {deepfake_likelihood * 100:.2f}%"""
    
    prompt = f"""
    You are an image verification specialist. Your task is to determine whether the provided image is part of a real-world event or a fake/edited scene.

    IMAGE DESCRIPTION:
    {description}
    
    REVERSE IMAGE SEARCH RESULTS:
    - Number of visually similar images found: {similar_images_count}
    - Visual match URLs: {', '.join(visual_match_urls)}
    - Visual match contents:
    {visual_match_contents_str}
    
    IMAGE ANALYSIS:
    - AI generation likelihood: {ai_generated_likelihood * 100:.2f}%{deepfake_info}
    
    CRITICAL RULES:
    - Reverse image search is a very important signal.
    - If the number of visually similar images is low, this could indicate fake news.
    - If the contents from the visual matches don't contain content similar to the image description, this is also likely to indicate fake news.
    - If contents mention "photoshopped", "not real", or similar phrases, that is strong evidence of fakery.
    - If AI generation likelihood is above 90%, consider this as strong evidence of potential fakery.
    - If the image is a portrait and deepfake likelihood is above 80%, consider this strong evidence of fakery.

    Format your response exactly as follows:
    CLASSIFICATION: [REAL or FAKE]
    CONFIDENCE: [LOW, MEDIUM, HIGH]
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
        sources=visual_match_urls
    )


    result_dict["decision"] = decision.model_dump()
    return result_dict


# Define the agent node that will decide which tools to use
def agent_node(state: ImageClassificationState) -> ImageClassificationState:
    
    # Create an agent with the tools
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Define the tools the agent can use
    tools = [
        analyze_image_description,
        check_ai_generation,
        check_deepfake,
        perform_reverse_image_search,
        make_classification_decision
    ]
    
    # Create the agent prompt with explicit instructions
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an image verification agent. Your task is to determine if an image is real or fake by using the appropriate tools.

        Follow this exact process:
        1. Analyze the image using analyze_image_description to get a detailed description and determine if it's a portrait
        - If analyzing with base64 fails, use the image_url as a fallback
        2. ALWAYS check if the image was AI-generated using check_ai_generation
        3. ALWAYS perform a reverse image search using perform_reverse_image_search
        4. ONLY if the image is a portrait, check for deepfakes using check_deepfake
        5. Make a final classification based on all the evidence using make_classification_decision

        IMPORTANT: When calling analyze_image_description, pass both the image_base64 AND the image_url as parameters to ensure successful analysis if one method fails."""),
                ("human", """I need to verify if this image is real or fake. 
        Image path: {image_path}
        Image URL: {image_url}

        Please analyze this image and determine if it's real or fake following the process I described. 
        Provide me with a detailed explanation of your reasoning and the URLs from the reverse image search you used to arrive at your conclusion.
        Only return the final classification, confidence, explanation, and sources.
        Don't describe the sources as "soruces from reverse image search", just describe them as "sources."""),
        ("ai", "{agent_scratchpad}")
    ])
    
    # Create the agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # Execute the agent
    result = agent_executor.invoke({
        "image_path": state["image"],
        "image_url": state["obs_url"],
        "image_base64": state["image_data"]["image_data"]
    })
    
    # Record the agent's reasoning
    state["agent_reasoning"] = result.get("output", "")
    
    # Update state with all the tools' results
    if "description" in result:
        state["description"] = result["description"]
    
    if "is_portrait" in result:
        state["is_portrait"] = result["is_portrait"]
    
    if "ai_generated_likelihood" in result:
        state["ai_generated_likelihood"] = result["ai_generated_likelihood"]
    
    if "deepfake_likelihood" in result:
        state["deepfake_likelihood"] = result["deepfake_likelihood"]
    
    if "similar_images_count" in result:
        state["similar_images_count"] = result["similar_images_count"]
    
    if "visual_match_urls" in result:
        state["visual_match_urls"] = result["visual_match_urls"]
    
    if "visual_match_contents" in result:
        state["visual_match_contents"] = result["visual_match_contents"]
    
    if "decision" in result:
        state["decision"] = result["decision"]
    
    return state


def create_image_classification_graph():
    workflow = StateGraph(ImageClassificationState)
    
    workflow.add_node("initialize", lambda state: {"sources": [], "visual_match_urls": [], "visual_match_contents": []})
    workflow.add_node("load_image", load_image)
    workflow.add_node("agent", agent_node)
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "load_image")
    workflow.add_edge("load_image", "agent")
    workflow.add_edge("agent", END)
    
    return workflow.compile()


graph = create_image_classification_graph()