import torch
import torch.nn as nn
import os
import sys
import importlib.util

# First, make sure all dependencies for BNext are available
try:
    import torch.nn.functional as F
    import numpy as np
    # These may not be installed yet - we'll handle them
    try:
        from einops import rearrange
    except ImportError:
        print("Installing einops...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "einops"])
        from einops import rearrange
        
    try:
        import timm
    except ImportError:
        print("Installing timm...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])
        from timm.models.layers import trunc_normal_, DropPath
    else:
        from timm.models.layers import trunc_normal_, DropPath
        
    try:
        import torchinfo
    except ImportError:
        print("Installing torchinfo...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torchinfo"])
except Exception as e:
    print(f"Error setting up dependencies: {e}")
    
# Define the path to the BNext model file directly
BNEXT_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BNext', 'src', 'bnext.py')
print(f"Loading BNext model from: {BNEXT_FILE_PATH}")

if os.path.exists(BNEXT_FILE_PATH):
    # Import BNext model using importlib
    try:
        spec = importlib.util.spec_from_file_location("bnext_module", BNEXT_FILE_PATH)
        bnext_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bnext_module)
        BNext = bnext_module.BNext
        print("Successfully imported BNext model")
    except Exception as e:
        print(f"Error importing BNext: {e}")
        raise ImportError(f"Failed to import BNext model: {e}")
else:
    raise FileNotFoundError(f"BNext model file not found at: {BNEXT_FILE_PATH}")

class BNextClassifier:
    def __init__(self, model_path, device=None):
        """
        Initialize BNext-L classifier for binary deepfake detection
        
        Args:
            model_path: Path to pretrained model weights (.pth.tar file)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Initialize model
        # For binary classification, we modify the original BNext-L model
        self.base_model = BNext(num_classes=1000, size="large")
        
        # Load pretrained weights from .pth.tar checkpoint for feature extraction
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            # If checkpoint contains 'state_dict' key
            state_dict = checkpoint['state_dict']
        else:
            # If checkpoint is already a state_dict
            state_dict = checkpoint
            
        # Remove 'module.' prefix if it exists (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                # Skip the final FC layer since we're replacing it
                if 'fc' not in k[7:]:
                    new_state_dict[k[7:]] = v
            else:
                if 'fc' not in k:
                    new_state_dict[k] = v
                
        # Load the cleaned state dict
        self.base_model.load_state_dict(new_state_dict, strict=False)
        
        # Replace the final classification layer for binary classification
        self.original_fc = self.base_model.fc
        feature_dim = self.original_fc.in_features
        self.base_model.fc = nn.Identity()  # Remove original fc layer
        
        # Create new binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # 2 classes: real and fake
        )
        
        # Move model to the appropriate device
        self.base_model = self.base_model.to(self.device)
        self.classifier = self.classifier.to(self.device)
        
        # Set to evaluation mode
        self.base_model.eval()
        self.classifier.eval()
        
        # Define class names
        self.class_names = ["REAL", "FAKE"]
    
    def classify(self, image_tensor):
        """
        Classify an image tensor as real or fake
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Dictionary with classification result and confidence
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            
            # Extract features from the base model
            features = self.base_model(image_tensor)
            
            # Pass features through our binary classifier
            outputs = self.classifier(features)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get the predicted class and confidence
            confidence, predicted_class = torch.max(probabilities, 1)
            
            result = {
                'class_id': predicted_class.item(),
                'class_name': self.class_names[predicted_class.item()],
                'confidence': confidence.item(),
                'is_fake': bool(predicted_class.item()),  # 0=real, 1=fake
                'full_probs': {
                    'real': probabilities[0, 0].item(),
                    'fake': probabilities[0, 1].item()
                }
            }
            
            return result