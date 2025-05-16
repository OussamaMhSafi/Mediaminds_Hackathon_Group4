import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import numpy as np

class ViTDeepFakeDetector:
    """
    A class to encapsulate the ViT-Base Transformer for DeepFake Detection
    by Prithiviraj Malli (prithivMLmods on Hugging Face)
    """
    
    def __init__(self, model_name="prithivMLmods/Deep-Fake-Detector-Model", device=None):
        """
        Initialize the ViT DeepFake Detector
        
        Args:
            model_name (str): The Hugging Face model identifier
            device (str, optional): Device to run inference on ('cpu' or specific device)
        """
        self.model_name = model_name
        
        # Use provided device or default to cpu
        self.device = device if device is not None else "cpu"
        
        # Initialize processor and model
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.model = ViTForImageClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set model to evaluation mode
    
    def predict(self, image_path, threshold=0.5):
        """
        Predict whether an image is real or fake
        
        Args:
            image_path (str or PIL.Image): Path to image or PIL Image object
            threshold (float): Confidence threshold for binary classification
            
        Returns:
            dict: Contains prediction probabilities, label, confidence score, and binary result
        """
        # Handle different input types
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, Image.Image):
            image = image_path
        else:
            raise TypeError("Input must be a file path or a PIL Image")
        
        # Model inference
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process outputs
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence, predicted_class_idx = torch.max(probabilities, dim=1)
        
        # Convert to usable results
        confidence = confidence.item()
        predicted_class_idx = predicted_class_idx.item()
        id2label = self.model.config.id2label
        label = id2label[predicted_class_idx]
        
        # Get class probabilities
        probs_dict = {
            id2label[idx].lower(): probabilities[0, idx].item() 
            for idx in range(probabilities.shape[1])
        }
        
        # Create standardized output format
        is_fake = label.lower() == "fake"
        
        # Ensure we have standardized keys for real/fake probabilities
        probabilities_dict = {
            "real": probs_dict.get("real", 1.0 - probs_dict.get("fake", 0.0)),
            "fake": probs_dict.get("fake", 1.0 - probs_dict.get("real", 0.0))
        }
        
        return {
            "probabilities": probabilities_dict,
            "label": label,
            "confidence": confidence,
            "is_fake": is_fake,
            "is_real": not is_fake,
            "above_threshold": confidence > threshold
        }
    
    def batch_predict(self, image_paths, batch_size=4, threshold=0.5):
        """
        Run prediction on multiple images
        
        Args:
            image_paths (list): List of image paths or PIL Images
            batch_size (int): Number of images to process at once
            threshold (float): Confidence threshold for binary classification
            
        Returns:
            list: List of prediction dictionaries with probabilities
        """
        results = []
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            
            # Load images
            batch_images = []
            for img in batch:
                if isinstance(img, str):
                    batch_images.append(Image.open(img).convert("RGB"))
                else:
                    batch_images.append(img)
            
            # Process batch
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get predictions
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidences, predicted_classes = torch.max(probabilities, dim=1)
            
            # Get id2label mapping
            id2label = self.model.config.id2label
            
            # Format results
            for j in range(len(batch)):
                confidence = confidences[j].item()
                predicted_class_idx = predicted_classes[j].item()
                label = id2label[predicted_class_idx]
                is_fake = label.lower() == "fake"
                
                # Get class probabilities
                probs_dict = {
                    id2label[idx].lower(): probabilities[j, idx].item() 
                    for idx in range(probabilities.shape[1])
                }
                
                # Ensure we have standardized keys for real/fake
                probabilities_dict = {
                    "real": probs_dict.get("real", 1.0 - probs_dict.get("fake", 0.0)),
                    "fake": probs_dict.get("fake", 1.0 - probs_dict.get("real", 0.0))
                }
                
                results.append({
                    "probabilities": probabilities_dict,
                    "label": label,
                    "confidence": confidence,
                    "is_fake": is_fake,
                    "is_real": not is_fake,
                    "above_threshold": confidence > threshold,
                    "image": batch[j] if isinstance(batch[j], str) else f"Image {i+j}"
                })
        
        return results


# Example usage
if __name__ == "__main__":
    # Create detector (CPU by default)
    detector = ViTDeepFakeDetector()
    
    # Or explicitly specify device
    # detector = ViTDeepFakeDetector(device="cuda:0")  # Use specific GPU
    
    # Single image prediction
    result = detector.predict("path/to/image.jpg")
    print(f"Image is {'fake' if result['is_fake'] else 'real'} with {result['confidence']:.2f} confidence")
    print(f"Probabilities: Real: {result['probabilities']['real']:.2f}, Fake: {result['probabilities']['fake']:.2f}")
    
    # Batch prediction (processes images in batches of 4 by default)
    batch_results = detector.batch_predict(["image1.jpg", "image2.jpg", "image3.jpg"])
    for i, res in enumerate(batch_results):
        print(f"Image {i+1}: {res['label']} ({res['confidence']:.2f})")