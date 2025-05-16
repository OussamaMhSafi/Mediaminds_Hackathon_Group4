# vlm_describer_graph.py
"""
Image describer using a Vision Language Model (VLM) with LangGraph.
Takes an image and a text prompt, and outputs a description.
"""
from typing import Dict, Any, Optional, Type
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from PIL import Image
import os
import sys
import time
import json

# Import LangGraph components
from langgraph.graph import START, END, StateGraph

# Import LangChain tool components
from langchain_core.tools import tool

# --- VLM Configuration ---
# Note: idefics2-8b-base is a large model. Ensure you have ~16GB VRAM for bfloat16 GPU,
# or >32GB RAM for CPU (will be slow). Consider smaller models for testing if needed.
VLM_MODEL_NAME = "HuggingFaceM4/idefics2-8b-base"
DEFAULT_PROMPT = "Describe this image in detail, focusing on the main subject and any notable actions or context."

# Import our VLM describer module
try:
    # This assumes smolvlm_image_describer.py is in the same directory orPYTHONPATH
    from smolvlm_image_describer import SmolVLMImageDescriber
except ImportError as e:
    print(f"Error importing SmolVLMImageDescriber module: {e}")
    print("Please ensure 'smolvlm_image_describer.py' is accessible and you have all VLM dependencies installed:")
    print("  pip install transformers torch Pillow requests accelerate")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import of VLM components: {e}")
    sys.exit(1)

# --- Graph Definition ---

# Define the output structure for the VLM description
class VLMDescriptionResult(BaseModel):
    model_name: str
    image_path_or_url: str
    prompt_used: str
    description: str
    error: Optional[str] = None # To store any error messages from VLM

# Define the state for the VLM description graph
class VLMDescriptionState(TypedDict):
    image_input_path: str  # Path or URL to the image
    prompt_text: str
    image_object: Optional[Any] # PIL.Image object
    raw_vlm_description: Optional[str] # Raw string output from VLM
    final_result: Optional[VLMDescriptionResult] # Pydantic model

# Functions for the graph nodes
def load_image_for_vlm(state: VLMDescriptionState) -> Dict[str, Any]:
    """Loads an image from the path or URL specified in the state."""
    image_path_or_url = state["image_input_path"]
    pil_image = None
    try:
        # SmolVLMImageDescriber can handle URL loading, but for consistency in graph,
        # we can load it here. However, SmolVLMImageDescriber's internal loading is robust.
        # Let's pass path/URL directly to VLM and store PIL for potential other uses.
        # For this graph, the VLM class will handle opening, so we mostly validate path.
        if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
            # For URLs, actual loading to PIL is deferred to VLM or can be done here if needed
            # For simplicity, we'll let the VLM handle URL fetching.
            # If we wanted to always have a PIL image in state['image_object']:
            # import requests
            # pil_image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert('RGB')
            pass # VLM will handle it
        elif os.path.exists(image_path_or_url):
            pil_image = Image.open(image_path_or_url).convert('RGB')
        else:
            raise FileNotFoundError(f"Image file not found at {image_path_or_url}")
        
        print(f"Image input '{image_path_or_url}' ready for VLM.")
        return {"image_object": pil_image} # Store PIL if local, None if URL (VLM handles URL)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise ValueError(f"Failed to access image: {e}") # Propagate error for graph handling
    except Exception as e:
        print(f"Error loading image from {image_path_or_url}: {e}")
        # It's better to raise here so the graph can potentially handle error states
        # or stop execution, rather than continuing with a missing image.
        raise ValueError(f"Failed to load image: {e}")


def run_vlm_describer(state: VLMDescriptionState) -> Dict[str, Any]:
    """Runs the VLM to describe the image using the provided prompt."""
    image_input = state["image_input_path"] # VLM class can take path/URL
    # Alternatively, if image_object is always populated with a PIL Image:
    # image_input = state.get("image_object")
    # if image_input is None and not state["image_input_path"].startswith("http"):
    #     raise ValueError("Image not loaded before running VLM describer.")
            
    prompt_text = state["prompt_text"]
    
    print(f"Running VLM ({VLM_MODEL_NAME}) on: {image_input[:50]}... with prompt: \"{prompt_text[:50]}...\"")
    try:
        # Initialize describer here. For frequent calls, consider pre-initializing.
        describer = SmolVLMImageDescriber(model_name=VLM_MODEL_NAME)
        start_time = time.time()
        # The SmolVLMImageDescriber.describe_image handles path, URL, or PIL image
        description = describer.describe_image(image_input, prompt_text=prompt_text)
        inference_time = time.time() - start_time
        print(f"VLM description completed in {inference_time:.2f}s.")
        
        if description.startswith("Error:"):
            print(f"VLM returned an error: {description}")
            # Store error in raw description to be handled by prepare_result
            return {"raw_vlm_description": description} 
        return {"raw_vlm_description": description}
    except Exception as e:
        print(f"Error during VLM description process: {e}")
        # Return an error message that can be part of the final result
        error_message = f"VLM describer failed: {str(e)}"
        return {"raw_vlm_description": f"Error: {error_message}"}


def prepare_vlm_result(state: VLMDescriptionState) -> Dict[str, Any]:
    """Formats the VLM's output into the VLMDescriptionResult Pydantic model."""
    raw_description = state.get("raw_vlm_description")
    image_path = state["image_input_path"]
    prompt = state["prompt_text"]

    if raw_description is None:
        desc = "Error: No description generated."
        err_msg = "VLM output was missing in state."
        print(f"Warning: {err_msg}")
    elif raw_description.startswith("Error:"):
        desc = "" # No successful description
        err_msg = raw_description 
        print(f"VLM process reported an error: {err_msg}")
    else:
        desc = raw_description
        err_msg = None

    result_model = VLMDescriptionResult(
        model_name=VLM_MODEL_NAME, # Or get from describer instance if it stores it
        image_path_or_url=image_path,
        prompt_used=prompt,
        description=desc,
        error=err_msg
    )
    return {"final_result": result_model}

# Create and compile the graph
def create_vlm_description_graph():
    workflow = StateGraph(VLMDescriptionState)
    workflow.add_node("load_image", load_image_for_vlm)
    workflow.add_node("run_vlm", run_vlm_describer)
    workflow.add_node("prepare_result", prepare_vlm_result)

    workflow.set_entry_point("load_image")
    workflow.add_edge("load_image", "run_vlm")
    workflow.add_edge("run_vlm", "prepare_result")
    workflow.add_edge("prepare_result", END)
    
    return workflow.compile()

compiled_vlm_describer_graph = create_vlm_description_graph()

# --- LangChain Tool Definition ---

class ImageDescriberToolInput(BaseModel):
    image_input_path: str = Field(description="The local file path or URL of the image to be described.")
    prompt_text: str = Field(description="The text prompt to guide the VLM's description.", default=DEFAULT_PROMPT)

@tool("visual_language_model_image_describer", args_schema=ImageDescriberToolInput, return_direct=False)
def image_describer_tool(image_input_path: str, prompt_text: str = DEFAULT_PROMPT) -> str:
    """
    Describes an image using a Vision Language Model (VLM) based on a given path/URL and prompt.
    Returns a JSON string with the description and related information, or an error message.
    """
    print(f"[Tool Call] image_describer_tool called with image: {image_input_path}, prompt: '{prompt_text}'")

    # Basic check for image path/URL presence, more robust checks in load_image node
    if not image_input_path:
        error_msg = "Error: Image path or URL cannot be empty."
        print(f"[Tool Error] {error_msg}")
        return error_msg
        
    try:
        initial_state: VLMDescriptionState = {
            "image_input_path": image_input_path,
            "prompt_text": prompt_text,
            "image_object": None,
            "raw_vlm_description": None,
            "final_result": None
        }
        
        final_state = compiled_vlm_describer_graph.invoke(initial_state)
        
        vlm_result: Optional[VLMDescriptionResult] = final_state.get("final_result")
        
        if vlm_result:
            if vlm_result.error:
                 print(f"[Tool Result] VLM process completed with an error: {vlm_result.error}")
            else:
                print(f"[Tool Result] VLM description successful for {image_input_path}")
            return vlm_result.model_dump_json()
        else:
            error_msg = "Error: VLM description pipeline did not produce a final result."
            print(f"[Tool Error] {error_msg}")
            # Fallback in case final_result is somehow None but no specific error was caught
            # Construct an error result to return
            error_result = VLMDescriptionResult(
                model_name=VLM_MODEL_NAME,
                image_path_or_url=image_input_path,
                prompt_used=prompt_text,
                description="",
                error=error_msg
            )
            return error_result.model_dump_json()
            
    except Exception as e:
        error_msg = f"Error during VLM image description for {image_input_path}: {str(e)}"
        print(f"[Tool Error] {error_msg} - Exception: {e}")
        # For critical errors not caught and formatted by the graph, return an error JSON
        error_result = VLMDescriptionResult(
            model_name=VLM_MODEL_NAME,
            image_path_or_url=image_input_path,
            prompt_used=prompt_text,
            description="",
            error=error_msg
        )
        return error_result.model_dump_json()

# --- Test function for direct graph invocation ---
def test_vlm_graph_direct(image_path_or_url: str, prompt: str) -> Optional[VLMDescriptionResult]:
    """Test function to run the VLM description pipeline directly using the compiled graph."""
    initial_state: VLMDescriptionState = {
        "image_input_path": image_path_or_url,
        "prompt_text": prompt,
        "image_object": None,
        "raw_vlm_description": None,
        "final_result": None
    }
    final_state = compiled_vlm_describer_graph.invoke(initial_state)
    return final_state.get("final_result")


# --- Main execution (for CLI testing) ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='VLM Image Describer with LangGraph')
    parser.add_argument('image_path_or_url', help='Path or URL to the image file/resource')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT, help='Prompt to guide the VLM description')
    parser.add_argument('--test_tool', action='store_true', help='Test the LangChain tool instead of direct graph invocation')
    args = parser.parse_args()
    
    # Simple validation for image_path_or_url for local files
    if not (args.image_path_or_url.startswith("http://") or args.image_path_or_url.startswith("https://")):
        if not os.path.exists(args.image_path_or_url):
            print(f"Error: Image file not found at {args.image_path_or_url}")
            sys.exit(1)

    if args.test_tool:
        print(f"\n--- TESTING VLM IMAGE DESCRIBER AS A LANGCHAIN TOOL ---")
        print(f"Processing image via tool: {args.image_path_or_url}")
        print(f"Using prompt: \"{args.prompt}\"")
        
        tool_input = {"image_input_path": args.image_path_or_url, "prompt_text": args.prompt}
        tool_output_json = image_describer_tool.invoke(tool_input)
        
        print("\nRaw Tool Output (JSON String):")
        print(tool_output_json)
        
        try:
            parsed_result = json.loads(tool_output_json)
            print("\nParsed Tool Output:")
            print(f"  Model Name: {parsed_result.get('model_name')}")
            print(f"  Image: {parsed_result.get('image_path_or_url')}")
            print(f"  Prompt: \"{parsed_result.get('prompt_used')}\"")
            if parsed_result.get('error'):
                print(f"  Error: {parsed_result.get('error')}")
            else:
                print(f"  Description: {parsed_result.get('description')}")
        except json.JSONDecodeError:
            print(f"\nTool output was not valid JSON (likely a direct error message string): {tool_output_json}")
        except Exception as e:
            print(f"\nError processing tool output: {e}")

    else:
        print(f"\n--- TESTING VLM IMAGE DESCRIBER GRAPH DIRECTLY ---")
        print(f"Processing image directly: {args.image_path_or_url}")
        print(f"Using prompt: \"{args.prompt}\"")
        
        result = test_vlm_graph_direct(args.image_path_or_url, args.prompt)
        
        if result:
            print("\n" + "=" * 60)
            print("VLM IMAGE DESCRIPTION RESULTS (DIRECT GRAPH INVOCATION)")
            print("=" * 60)
            print(f"\nImage Source: {result.image_path_or_url}")
            print(f"Model: {result.model_name}")
            print(f"Prompt Used: \"{result.prompt_used}\"")
            if result.error:
                print(f"\nError during processing: {result.error}")
            else:
                print(f"\n📝 Generated Description:")
                print(result.description)
            print("\n" + "=" * 60)
        else:
            print("VLM description pipeline failed or produced no result.")

# To make the graph easily importable
vlm_graph = compiled_vlm_describer_graph