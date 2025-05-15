"""
Simplified deepfake detector using BNext-L with LangGraph
Takes an image and outputs a classification, without web search
"""
from typing import Dict, Any, Optional, Literal, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel
import operator
from PIL import Image
import os
import sys

# Import LangGraph components
from langgraph.graph import START, END, StateGraph

# Import custom modules for BNext-L
import importlib.util

# Define the path to the BNext model file directly
BNEXT_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BNext', 'src', 'bnext.py')

# Import BNext model using importlib
try:
    # Try importing any required dependencies
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import numpy as np
    except ImportError as e:
        print(f"Error importing PyTorch dependencies: {e}")
        sys.exit(1)
        
    # These may need to be installed
    try:
        from einops import rearrange
    except ImportError:
        print("Installing einops...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "einops"])
        from einops import rearrange
        
    try:
        import timm
        from timm.models.layers import trunc_normal_, DropPath
    except ImportError:
        print("Installing timm...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])
        from timm.models.layers import trunc_normal_, DropPath
        
    try:
        import torchvision
    except ImportError:
        print("Installing torchvision...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision"])
        import torchvision
        
    # Load the BNext model
    print(f"Loading BNext model from: {BNEXT_FILE_PATH}")
    
    if os.path.exists(BNEXT_FILE_PATH):
        spec = importlib.util.spec_from_file_location("bnext_module", BNEXT_FILE_PATH)
        bnext_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bnext_module)
        BNext = bnext_module.BNext
        print("Successfully imported BNext model")
    else:
        print(f"Error: BNext model file not found at {BNEXT_FILE_PATH}")
        sys.exit(1)
except Exception as e:
    print(f"Error importing BNext model: {e}")
    sys.exit(1)

# Define the output structure
class DeepfakeResult(BaseModel):
    classification: Literal["REAL", "FAKE"]
    confidence: float
    probabilities: Dict[str, float]

# Define the state which will be passed around
class DeepfakeDetectionState(TypedDict):
    image_path: str
    image_tensor: Optional[Any]
    result: Optional[DeepfakeResult]

# Image preprocessing function
def preprocess_image(image_path):
    """Preprocess image for BNext-L model input"""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

class BNextClassifier:
    """Simplified BNext classifier for binary deepfake detection"""
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
            
        # Initialize model for binary classification
        self.base_model = BNext(num_classes=1000, size="large")
        
        # Load pretrained weights
        print(f"Loading model weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if it exists
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                if 'fc' not in k[7:]:
                    new_state_dict[k[7:]] = v
            else:
                if 'fc' not in k:
                    new_state_dict[k] = v
                    
        print(f"Loaded {len(new_state_dict)} layers from checkpoint")
        
        # Load the cleaned state dict
        self.base_model.load_state_dict(new_state_dict, strict=False)
        
        # Replace the final classification layer for binary classification
        self.original_fc = self.base_model.fc
        feature_dim = self.original_fc.in_features
        self.base_model.fc = torch.nn.Identity()  # Remove original fc layer
        
        # Create new binary classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 2)  # 2 classes: real and fake
        )
        
        # Move model to device
        self.base_model = self.base_model.to(self.device)
        self.classifier = self.classifier.to(self.device)
        
        # Set to evaluation mode
        self.base_model.eval()
        self.classifier.eval()
        
        # Class names
        self.class_names = ["REAL", "FAKE"]
    
    def classify(self, image_tensor):
        """Classify an image tensor as real or fake"""
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            
            # Extract features from the base model
            features = self.base_model(image_tensor)
            
            # Pass features through binary classifier
            outputs = self.classifier(features)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get predicted class and confidence
            confidence, predicted_class = torch.max(probabilities, 1)
            
            result = {
                'classification': self.class_names[predicted_class.item()],
                'confidence': confidence.item(),
                'probabilities': {
                    'real': probabilities[0, 0].item(),
                    'fake': probabilities[0, 1].item()
                }
            }
            
            return result

# Initialize the classifier with model path
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'bnext_l_pretrained.pth.tar')
if not os.path.exists(MODEL_PATH):
    print(f"Warning: Model file not found at {MODEL_PATH}")
    print("The graph will be created but will fail at runtime until the model file is available")

# Functions for the graph nodes
def load_image(state):
    """Load and preprocess the image"""
    image_path = state["image_path"]
    image_tensor = preprocess_image(image_path)
    return {"image_tensor": image_tensor}

def classify_image(state):
    """Classify the image using BNext-L"""
    # Initialize classifier only when needed (to avoid loading model during import)
    classifier = BNextClassifier(model_path=MODEL_PATH)
    image_tensor = state["image_tensor"]
    
    # Get classification result
    result_dict = classifier.classify(image_tensor)
    
    # Create the result object
    result = DeepfakeResult(
        classification=result_dict['classification'],
        confidence=result_dict['confidence'],
        probabilities=result_dict['probabilities']
    )
    
    return {"result": result}

# Create the graph
def create_deepfake_detection_graph():
    """Create the LangGraph workflow for deepfake detection"""
    workflow = StateGraph(DeepfakeDetectionState)
    
    # Add nodes
    workflow.add_node("load_image", load_image)
    workflow.add_node("classify_image", classify_image)
    
    # Define the edges between nodes (sequential flow)
    workflow.add_edge(START, "load_image")
    workflow.add_edge("load_image", "classify_image")
    workflow.add_edge("classify_image", END)
    
    # Compile the graph
    return workflow.compile()

# Create the graph, to launch using 'langgraph dev'
graph = create_deepfake_detection_graph()

def test_detection(image_path):
    """Test function to run the detection pipeline directly"""
    result = graph.invoke({"image_path": image_path})
    return result["result"]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='BNext-L Deepfake Detection with LangGraph')
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
    print(f"Classification: {result.classification}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Real probability: {result.probabilities['real']:.4f}")
    print(f"Fake probability: {result.probabilities['fake']:.4f}")
    
    # Print final determination
    print("\n🔍 FINAL DETERMINATION:")
    is_fake = result.classification == "FAKE"
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