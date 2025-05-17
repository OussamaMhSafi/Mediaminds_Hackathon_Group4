# image_processing_utils.py (selected, adapted functions)
# (Keep all your original imports: Dict, Any, Optional, etc.)
from typing import Dict, Any, Optional, List, Literal, Annotated
from typing_extensions import TypedDict
from langchain_community.document_loaders import WebBaseLoader
from pydantic import BaseModel, Field
from enum import Enum
import operator # Not strictly needed for agent tools if not appending state
import base64
from io import BytesIO
from PIL import Image
import re
import os
import uuid
from collections import Counter # Not used in this refactor
import requests
from serpapi import search as GoogleSearch
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI # Keep for describe_image if it uses it
from obs import ObsClient # Keep for obs_upload_and_get_url
import json
from langchain.schema import HumanMessage
# Keep your Decision BaseModel
class Decision(BaseModel):
    classification: Literal["REAL", "FAKE"]
    confidence: int
    explanation: str
    sources: List[str]

# Adapted/New utility functions
def obs_upload_and_get_url(image_path: str, image_obj: Image.Image) -> Optional[str]:
    try:
        ak = os.getenv("AccessKeyID")
        sk = os.getenv("SecretAccessKey")
        server = "https://obs.ap-southeast-3.myhuaweicloud.com" # Your server
        obsClient = ObsClient(access_key_id=ak, secret_access_key=sk, server=server)
        
        file_name = os.path.basename(image_path)
        unique_id = str(uuid.uuid4())[:8]
        object_key = f"images/{unique_id}_{file_name}" # Your bucket path
        bucket_name = "groupd" # Your bucket name
        
        resp = obsClient.putFile(bucket_name, object_key, image_path)
        
        if resp.status < 300:
            endpoint = "obs.ap-southeast-3.myhuaweicloud.com" # Your endpoint
            public_url = f"https://{bucket_name}.{endpoint}/{object_key}"
            print(f"OBS Upload succeeded: {public_url}")
            return public_url
        else:
            print(f"OBS Upload failed with status {resp.status}")
            return None
    except Exception as e:
        print(f"Error during OBS upload: {str(e)}")
        return None

def get_image_base64(image_obj: Image.Image) -> str:
    buffered = BytesIO()
    image_obj.save(buffered, format=image_obj.format or "JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_image_description_from_base64(image_base64: str, llm) -> str: # Pass LLM
    messages = [
        HumanMessage(content=[
            {"type": "text", "text": "Describe this image focusing on verifiable elements like people, location, text, events, objects, and date indicators."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ])
    ]
    response = llm.invoke(messages)
    return response.content

def get_ai_likelihood_from_path(image_path: str) -> float:
    try:
        params = {
            'models': 'genai',
            'api_user': os.getenv("SIGHTENGINE_API_USER"), 
            'api_secret': os.getenv("SIGHTENGINE_API_SECRET") 
        }
        files = {'media': open(image_path, 'rb')}
        response = requests.post('https://api.sightengine.com/1.0/check.json', files=files, data=params)
        result = json.loads(response.text)
        if result.get('status') == 'success':
            return result.get('type', {}).get('ai_generated', 0.0)
        else:
            print(f"SightEngine API error (AI Gen): {result}")
            return 0.0
    except Exception as e:
        print(f"Error in AI generation detection util: {str(e)}")
        return 0.0

# This tool can be imported directly as it's already well-defined
@tool
def check_deepfake_from_url(url: str) -> Dict[str, Any]:
    """
    Checks if a portrait image (URL) contains a deepfake using Sightengine API.
    Only use this for close-up portraits of a single person's face.
    Args: url: URL of the image to check
    Returns: Dictionary with deepfake detection results including 'deepfake_likelihood'.
    """
    try:
        params = {
            'url': url,
            'models': 'deepfake',
            'api_user': os.getenv("SIGHTENGINE_API_USER"),
            'api_secret': os.getenv("SIGHTENGINE_API_SECRET")
        }
        response = requests.get('https://api.sightengine.com/1.0/check.json', params=params)
        result = json.loads(response.text)
        if result.get('status') == 'success':
            return {
                "success": True,
                "deepfake_likelihood": result.get('type', {}).get('deepfake', 0.0),
                "raw_response": result
            }
        else:
            error_msg = result.get('error', {}).get('message', 'Unknown error')
            return {"success": False, "error": f"API error: {error_msg}", "deepfake_likelihood": 0.0}
    except Exception as e:
        return {"success": False, "error": str(e), "deepfake_likelihood": 0.0}


def get_reverse_image_search_results(image_url: str) -> Dict[str, Any]:
    try:
        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": os.getenv("SERPAPI_API_KEY") # Ensure SERPAPI_API_KEY is set
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        
        visual_matches = results.get("visual_matches", [])
        similar_images_count = len(visual_matches)
        
        top_matches = visual_matches[:5] # Limit to top 5 for brevity in agent context
        visual_match_urls = [match.get("link", "") for match in top_matches if match.get("link")]
        
        # Fetching content can be slow and verbose for an agent's observation.
        # For a minimal version, we might just return URLs and titles.
        # Or, the agent can be prompted to ask for content of specific URLs if needed.
        # For now, let's keep it simple and return URLs and maybe titles if available.
        visual_match_details = []
        for match in top_matches:
            detail = {"url": match.get("link"), "title": match.get("title"), "source": match.get("source")}
            if detail["url"]:
                 visual_match_details.append(detail)
        
        return {
            "similar_images_count": similar_images_count,
            "visual_matches": visual_match_details # List of dicts with url, title, source
        }
    except Exception as e:
        print(f"Error in reverse image search util: {str(e)}")
        return {
            "similar_images_count": 0,
            "visual_matches": [],
            "error": str(e)
        }
# Ensure other necessary imports from your original file are here for these utils to work
# e.g., PIL, base64, BytesIO, requests, SerpAPI, ObsClient, os, uuid, json
# Also, make sure API keys are loaded from environment variables (os.getenv)
# SIGHTENGINE_API_USER, SIGHTENGINE_API_SECRET, SERPAPI_API_KEY, AccessKeyID, SecretAccessKey