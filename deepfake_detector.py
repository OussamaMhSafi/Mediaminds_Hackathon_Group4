"""
Simple deepfake detector using ViT transformer model with LangGraph
Takes an image and outputs classification
"""
from typing import Dict, Any, Optional, Type
from typing_extensions import TypedDict
from pydantic import BaseModel, Field # Added Field
from PIL import Image
import os
import sys
import time

# Import LangGraph components
from langgraph.graph import START, END, StateGraph

# Import LangChain tool components
from langchain_core.tools import tool # Use the decorator

# Import our transformer detector module
try:
    import torch
    from vit_deepfake_detector import ViTDeepFakeDetector
except ImportError as e:
    print(f"Error importing transformer detector module: {e}")
    print("Installing required dependencies...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_transformer.txt"])
        import torch
        print("Trying to import detector module after installing dependencies...")
        from vit_deepfake_detector import ViTDeepFakeDetector
    except Exception as e:
        print(f"Error setting up transformer detector: {e}")
        sys.exit(1)

# --- Existing Graph Code (mostly unchanged) ---

# Define the output structure
class ModelResult(BaseModel):
    model_name: str
    label: str
    confidence: float
    probabilities: Dict[str, float]
    is_fake: bool
    model_dict: Dict[str, Any] = {}

# Define the state which will be passed around
class DeepfakeDetectionState(TypedDict):
    image_path: str
    image: Optional[Any]
    vit_result: Optional[Dict[str, Any]]
    result: Optional[ModelResult]

# Functions for the graph nodes
def load_image(state: DeepfakeDetectionState) -> Dict[str, Any]:
    image_path = state["image_path"]
    try:
        pil_image = Image.open(image_path).convert('RGB')
        return {"image": pil_image}
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        raise ValueError(f"Failed to load image: {e}")

def run_vit_detector(state: DeepfakeDetectionState) -> Dict[str, Any]:
    pil_image = state.get("image")
    if pil_image is None:
        raise ValueError("Image not loaded before running ViT detector.")
    try:
        detector = ViTDeepFakeDetector()
        start_time = time.time()
        result = detector.predict(pil_image)
        inference_time = time.time() - start_time
        print(f"ViT detector completed in {inference_time:.2f}s")
        return {"vit_result": result}
    except Exception as e:
        print(f"Error in ViT detector: {e}")
        raise ValueError(f"ViT detector failed: {e}")

def prepare_result(state: DeepfakeDetectionState) -> Dict[str, Any]:
    vit_result = state.get("vit_result")
    if vit_result is None:
        print("Warning: vit_result not found in state for prepare_result.")
        return {}
    model_result = ModelResult(
        model_name="ViT",
        label=vit_result["label"],
        confidence=vit_result["confidence"],
        probabilities=vit_result["probabilities"],
        is_fake=vit_result["is_fake"],
        model_dict=vit_result
    )
    return {"result": model_result}

# Create the graph
def create_deepfake_detection_graph():
    workflow = StateGraph(DeepfakeDetectionState)
    workflow.add_node("load_image", load_image)
    workflow.add_node("run_vit_detector", run_vit_detector)
    workflow.add_node("prepare_result", prepare_result)
    workflow.set_entry_point("load_image")
    workflow.add_edge("load_image", "run_vit_detector")
    workflow.add_edge("run_vit_detector", "prepare_result")
    workflow.add_edge("prepare_result", END)
    return workflow.compile()

compiled_deepfake_detector_graph = create_deepfake_detection_graph()

class DeepfakeDetectorToolInput(BaseModel):
    image_path: str = Field(description="The local file path to the image to be analyzed for deepfakes.")

@tool("deepfake_image_analyzer", args_schema=DeepfakeDetectorToolInput, return_direct=False)
def deepfake_image_analyzer_tool(image_path: str) -> str:
    """
    Analyzes an image from the given local file path to detect if it's a deepfake.
    Returns a JSON string with the classification (real/fake), confidence, and probabilities.
    If an error occurs (e.g., file not found, detection error), an error message string is returned.
    """
    print(f"[Tool Call] deepfake_image_analyzer_tool called with image_path: {image_path}")
    if not os.path.exists(image_path):
        error_msg = f"Error: Image file not found at {image_path}"
        print(f"[Tool Error] {error_msg}")
        return error_msg
    
    try:
        # Prepare initial state for the graph
        # All keys from DeepfakeDetectionState must be present
        initial_state: DeepfakeDetectionState = {
            "image_path": image_path,
            "image": None,
            "vit_result": None,
            "result": None
        }
        
        # Invoke the pre-compiled graph
        final_state = compiled_deepfake_detector_graph.invoke(initial_state)
        
        detection_result: Optional[ModelResult] = final_state.get("result")
        
        if detection_result:
            print(f"[Tool Result] Detection successful: {detection_result.label}")
            return detection_result.model_dump_json() # Return Pydantic model as JSON string
        else:
            error_msg = "Error: Deepfake detection pipeline did not produce a result."
            print(f"[Tool Error] {error_msg}")
            return error_msg
            
    except Exception as e:
        error_msg = f"Error during deepfake detection for {image_path}: {str(e)}"
        print(f"[Tool Error] {error_msg} - Exception: {e}")
        # For debugging, you might want to log the full traceback here
        return error_msg

# --- Test function for the graph (can be kept for direct testing) ---
def test_detection_graph_direct(image_path: str) -> Optional[ModelResult]:
    """Test function to run the detection pipeline directly using the compiled graph"""
    initial_state: DeepfakeDetectionState = {
        "image_path": image_path,
        "image": None,
        "vit_result": None,
        "result": None
    }
    final_state = compiled_deepfake_detector_graph.invoke(initial_state)
    return final_state.get("result")


# --- Main execution (for CLI testing of the graph or tool) ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ViT-based Deepfake Detection with LangGraph')
    parser.add_argument('image_path', help='Path to the image file to classify')
    parser.add_argument('--test_tool', action='store_true', help='Test the LangChain tool instead of direct graph invocation')
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        sys.exit(1)
    
    if args.test_tool:
        print(f"\n--- TESTING DEEPFAKE DETECTOR AS A LANGCHAIN TOOL ---")
        print(f"Processing image via tool: {args.image_path}")
        tool_output_json = deepfake_image_analyzer_tool.invoke({"image_path": args.image_path}) # Tool expects a dict matching args_schema
        print("\nRaw Tool Output (JSON String):")
        print(tool_output_json)
        
        # Try to parse the JSON to display it nicely (optional, for demo)
        try:
            # Assuming the output is JSON from ModelResult or an error string
            if tool_output_json.startswith("Error:"):
                print(f"\nTool returned an error: {tool_output_json}")
            else:
                import json
                parsed_result = json.loads(tool_output_json)
                print("\nParsed Tool Output:")
                print(f"  Model Name: {parsed_result.get('model_name')}")
                print(f"  Label: {parsed_result.get('label')}")
                print(f"  Confidence: {parsed_result.get('confidence'):.4f}")
                print(f"  Is Fake: {parsed_result.get('is_fake')}")
                print(f"  Probabilities: {parsed_result.get('probabilities')}")
        except json.JSONDecodeError:
            print(f"\nTool output was not valid JSON (likely an error message): {tool_output_json}")
        except Exception as e:
            print(f"\nError processing tool output: {e}")

    else:
        print(f"\n--- TESTING DEEPFAKE DETECTOR GRAPH DIRECTLY ---")
        print(f"Processing image directly: {args.image_path}")
        result = test_detection_graph_direct(args.image_path)
        
        if result:
            print("\n" + "=" * 60)
            print("DEEPFAKE DETECTION RESULTS (DIRECT GRAPH INVOCATION)")
            print("=" * 60)
            print(f"\nImage: {args.image_path}")
            print("\n⭐ ViT MODEL RESULTS:")
            print(f"  Classification: {result.label}")
            print(f"  Confidence: {result.confidence:.4f}")
            print(f"  Real probability: {result.probabilities['real']:.4f}")
            print(f"  Fake probability: {result.probabilities['fake']:.4f}")
            
            width = 40
            print("\n📊 PROBABILITY VISUALIZATION:")
            real_chars = int(result.probabilities['real'] * width)
            fake_chars = int(result.probabilities['fake'] * width)
            print(f"  REAL: {'█' * real_chars}{' ' * (width - real_chars)} {result.probabilities['real']:.2f}")
            print(f"  FAKE: {'█' * fake_chars}{' ' * (width - fake_chars)} {result.probabilities['fake']:.2f}")
            
            print("\n🔍 FINAL DETERMINATION:")
            is_fake = result.is_fake
            confidence = result.confidence
            if is_fake and confidence >= 0.7: print(f"  ❌ HIGH CONFIDENCE FAKE ({confidence:.2f})")
            elif is_fake and confidence >= 0.4: print(f"  ⚠️ LIKELY FAKE ({confidence:.2f})")
            elif not is_fake and confidence >= 0.7: print(f"  ✅ HIGH CONFIDENCE REAL ({confidence:.2f})")
            elif not is_fake and confidence >= 0.4: print(f"  ℹ️ LIKELY REAL ({confidence:.2f})")
            else: print(f"  ❓ UNCERTAIN ({confidence:.2f})")
            print("\n" + "=" * 60)
        else:
            print("Detection failed or produced no result.")


graph = compiled_deepfake_detector_graph