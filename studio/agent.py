# image_classifier_agent.py

import os
from dotenv import load_dotenv
from typing import Dict, Any, List, Literal
from PIL import Image

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool, ToolException
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.messages import HumanMessage, AIMessage

# Import reusable components from your utility file
from new_utils import (
    Decision, # Pydantic model for final output
    obs_upload_and_get_url,
    get_image_base64,
    get_image_description_from_base64,
    get_ai_likelihood_from_path,
    check_deepfake_from_url as official_check_deepfake_tool, # Already a @tool
    get_reverse_image_search_results
)
import json
# Load environment variables (ensure you have .env file or vars are set)
load_dotenv()
# Required: OPENAI_API_KEY, AccessKeyID, SecretAccessKey, SIGHTENGINE_API_USER, SIGHTENGINE_API_SECRET, SERPAPI_API_KEY

# --- Agent Tools ---

@tool
def get_initial_image_analysis(image_path: str) -> Dict[str, Any]:
    """
    Loads an image from a local path, uploads it to cloud storage (OBS),
    describes its content using an LLM, and detects AI generation likelihood.
    This should be the first tool called.
    Returns a dictionary with 'obs_url', 'description', 'ai_generated_likelihood',
    and 'error' if any step failed.
    """
    try:
        img = Image.open(image_path)
        img_format = img.format # Store format before it's lost
    except Exception as e:
        raise ToolException(f"Failed to load image from path: {image_path}. Error: {str(e)}")

    # Ensure image object has format for saving if not originally present
    if not img.format:
        img.format = "JPEG" # Default to JPEG if format is missing

    image_base64_str = get_image_base64(img)
    
    # OBS Upload
    public_obs_url = obs_upload_and_get_url(image_path, img)
    if not public_obs_url:
        # Continue even if OBS fails, some tools might not need it or can use base64
        print(f"Warning: OBS upload failed for {image_path}. Some functionalities might be limited.")
        # return {"error": "OBS upload failed."} # Or raise ToolException

    # Image Description (requires an LLM instance for this tool)
    # For simplicity, we'll initialize a dedicated one here or pass it if this tool becomes a class method
    desc_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # Or gpt-4-vision-preview
    description = get_image_description_from_base64(image_base64_str, desc_llm)
    
    # AI Generation Detection
    ai_likelihood = get_ai_likelihood_from_path(image_path)
    
    return {
        "obs_url": public_obs_url,
        "description": description,
        "ai_generated_likelihood": ai_likelihood,
        "image_path_for_reference": image_path # In case other tools need it
    }

@tool
def perform_reverse_image_search_on_url(image_url: str) -> Dict[str, Any]:
    """
    Performs a reverse image search using Google Lens (via SerpAPI) for the given image URL.
    Returns a dictionary with 'similar_images_count' and 'visual_matches' (list of dicts with url, title, source).
    """
    if not image_url:
        raise ToolException("Image URL is required for reverse image search.")
    return get_reverse_image_search_results(image_url)

# The official_check_deepfake_tool is imported and used directly.

# --- Agent Setup ---
def create_image_classifier_agent():
    # Define the LLM for the agent
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # Or "gpt-4o"

    # List of tools available to the agent
    tools = [
        get_initial_image_analysis,
        official_check_deepfake_tool, # Renamed at import for clarity
        perform_reverse_image_search_on_url
    ]

    # Agent prompt
    # Note: The 'input' variable in the prompt will be the user's query (image path).
    # The 'agent_scratchpad' will be filled by the agent framework with tool calls and observations.
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert image forensics analyst. Your goal is to determine if an image is REAL or FAKE.
You must use the available tools to gather evidence and then make a reasoned classification.

Follow these steps:
1.  Use the `get_initial_image_analysis` tool with the provided image path. This will give you an OBS URL, a description of the image, and an AI generation likelihood.
2.  Review the image 'description'. If it clearly describes a close-up portrait of a single person's face, you *may* use the `official_check_deepfake_tool` with the 'obs_url' obtained in step 1. Do NOT use it for other types of images. If you use it, consider its 'deepfake_likelihood' output.
3.  Use the `perform_reverse_image_search_on_url` tool with the 'obs_url'. Analyze its 'similar_images_count' and 'visual_matches'.
4.  Synthesize all gathered information:
    - Image description
    - AI generation likelihood (from `get_initial_image_analysis`)
    - Deepfake likelihood (if `official_check_deepfake_tool` was used)
    - Reverse image search results (similar images count, URLs and titles of matches). Pay close attention if `similar_images_count` is low (e.g., < 5-10), or if visual matches point to stock photo sites, or sites known for manipulated content.
5.  Provide your final decision in the following JSON format ONLY:
    ```json
    {{
        "classification": "REAL or FAKE",
        "confidence": <integer between 0 and 100>,
        "explanation": "Detailed reasoning based on the evidence. Mention key findings from each tool used and how they contributed to your decision. Cite specific source URLs from visual matches if they are relevant to your explanation.",
        "sources": ["list", "of", "relevant", "supporting", "URLs", "from", "visual_matches"]
    }}
    ```
If any tool call fails or returns an error, acknowledge it and make your best judgment with the available information.
If essential information (like OBS URL for further checks) is missing due to an error in the initial tool, state that you cannot proceed fully.
"""),
        ("user", "{input}"), # The user's query (image path)
        MessagesPlaceholder(variable_name="agent_scratchpad") # For agent's intermediate steps
    ])

    # Create the agent
    agent = create_openai_functions_agent(llm, tools, prompt_template)

    # Create the AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

# --- Main execution ---
if __name__ == "__main__":
    # Ensure your .env file has:
    # OPENAI_API_KEY="your_openai_key"
    # AccessKeyID="your_huawei_obs_ak"
    # SecretAccessKey="your_huawei_obs_sk"
    # SIGHTENGINE_API_USER="your_sightengine_user"
    # SIGHTENGINE_API_SECRET="your_sightengine_secret"
    # SERPAPI_API_KEY="your_serpapi_key"

    # Create the agent
    image_agent = create_image_classifier_agent()
    
    # Example usage:
    # Replace with the actual path to your test image
    # test_image_path = "path/to/your/image.jpg"
    # test_image_path = "test_images/real_news_hurricane.jpeg"
    test_image_path = r'C:\Users\emran\repos\Mediaminds_Hackathon_Group4\images\image1.jpg'

    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}")
    else:
        print(f"Analyzing image: {test_image_path}")
        try:
            # The input to the agent is a dictionary, as expected by create_openai_functions_agent
            response = image_agent.invoke({"input": test_image_path})
            print("\n--- Agent's Final Output ---")
            # The actual decision should be in response['output']
            # It might be a string containing JSON, so parse it if necessary.
            try:
                # The agent is prompted to return JSON, so the output should be a stringified JSON
                final_decision = json.loads(response['output'])
                print(json.dumps(final_decision, indent=2))
            except json.JSONDecodeError:
                print("Output was not valid JSON, printing as is:")
                print(response['output'])
            except TypeError: # If output is already a dict
                 if isinstance(response['output'], dict):
                    print(json.dumps(response['output'], indent=2))
                 else:
                    print("Output was not a dict or JSON string, printing as is:")
                    print(response['output'])


        except Exception as e:
            print(f"An error occurred during agent execution: {e}")
            import traceback
            traceback.print_exc()