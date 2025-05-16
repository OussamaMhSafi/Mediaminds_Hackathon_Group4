"""
Simple deepfake detector using ViT transformer model with LangGraph
Takes an image and outputs classification
"""
from typing import Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel
from PIL import Image
import os
import sys
import time

# Import LangGraph components
from langgraph.graph import START, END, StateGraph

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
    image: Optional[Any]  # MODIFIED: Was Optional[Image.Image]
    vit_result: Optional[Dict[str, Any]]
    result: Optional[ModelResult]
# Functions for the graph nodes
def load_image(state):
    """Load the image for processing"""
    image_path = state["image_path"]
    
    try:
        # Open image and convert to RGB
        image = Image.open(image_path).convert('RGB')
        # Return both the original state and the new image
        return {**state, "image": image}
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        raise ValueError(f"Failed to load image: {e}")

def run_vit_detector(state):
    """Run the ViT-based detector on the image"""
    image = state["image"]
    
    try:
        # Initialize the ViT detector
        detector = ViTDeepFakeDetector()
        
        # Run prediction
        start_time = time.time()
        result = detector.predict(image)
        inference_time = time.time() - start_time
        
        print(f"ViT detector completed in {inference_time:.2f}s")
        
        return {**state, "vit_result": result}
    except Exception as e:
        print(f"Error in ViT detector: {e}")
        raise ValueError(f"ViT detector failed: {e}")

def prepare_result(state):
    """Format the ViT result into a ModelResult"""
    vit_result = state.get("vit_result")
    
    # If no result yet, return original state
    if vit_result is None:
        return state
    
    # Create model result object
    model_result = ModelResult(
        model_name="ViT",
        label=vit_result["label"],
        confidence=vit_result["confidence"],
        probabilities=vit_result["probabilities"],
        is_fake=vit_result["is_fake"],
        model_dict=vit_result  # Store the full result dictionary
    )
    
    # Return the original state plus the result
    return {**state, "result": model_result}

# Create the graph
def create_deepfake_detection_graph():
    """Create the LangGraph workflow for deepfake detection"""
    workflow = StateGraph(DeepfakeDetectionState)
    
    # Add nodes
    workflow.add_node("load_image", load_image)
    workflow.add_node("run_vit_detector", run_vit_detector)
    workflow.add_node("prepare_result", prepare_result)
    
    # Create a sequential workflow
    # First load the image
    workflow.add_edge(START, "load_image")
    
    # After loading, run detector
    workflow.add_edge("load_image", "run_vit_detector")
    workflow.add_edge("run_vit_detector", "prepare_result")
    
    # Exit after preparing result
    workflow.add_edge("prepare_result", END)
    
    # Compile the graph
    return workflow.compile()

# Create the graph, to launch using 'langgraph dev'
graph = create_deepfake_detection_graph()

def test_detection(image_path):
    """Test function to run the detection pipeline directly"""
    # Create the initial state with the image path
    initial_state = {
        "image_path": image_path
    }
    result = graph.invoke(initial_state)
    return result["result"]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ViT-based Deepfake Detection with LangGraph')
    parser.add_argument('image_path', help='Path to the image file to classify')
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        sys.exit(1)
    
    # Run the detection
    print(f"Processing image: {args.image_path}")
    result = test_detection(args.image_path)
    
    # Print results
    print("\n" + "=" * 60)
    print("DEEPFAKE DETECTION RESULTS")
    print("=" * 60)
    
    print(f"\nImage: {args.image_path}")
    
    # ViT results
    print("\n⭐ ViT MODEL RESULTS:")
    print(f"  Classification: {result.label}")
    print(f"  Confidence: {result.confidence:.4f}")
    print(f"  Real probability: {result.probabilities['real']:.4f}")
    print(f"  Fake probability: {result.probabilities['fake']:.4f}")
    
    # Print visual representation of probabilities
    width = 40
    print("\n📊 PROBABILITY VISUALIZATION:")
    
    # ViT
    real_chars = int(result.probabilities['real'] * width)
    fake_chars = int(result.probabilities['fake'] * width)
    print(f"  REAL: {'█' * real_chars}{' ' * (width - real_chars)} {result.probabilities['real']:.2f}")
    print(f"  FAKE: {'█' * fake_chars}{' ' * (width - fake_chars)} {result.probabilities['fake']:.2f}")
    
    # Print final determination
    print("\n🔍 FINAL DETERMINATION:")
    is_fake = result.is_fake
    confidence = result.confidence
    
    if is_fake and confidence >= 0.7:
        print(f"  ❌ HIGH CONFIDENCE FAKE ({confidence:.2f})")
    elif is_fake and confidence >= 0.4:
        print(f"  ⚠️ LIKELY FAKE ({confidence:.2f})")
    elif not is_fake and confidence >= 0.7:
        print(f"  ✅ HIGH CONFIDENCE REAL ({confidence:.2f})")
    elif not is_fake and confidence >= 0.4:
        print(f"  ℹ️ LIKELY REAL ({confidence:.2f})")
    else:
        print(f"  ❓ UNCERTAIN ({confidence:.2f})")
    
    print("\n" + "=" * 60)
