"""
Simple script to run the deepfake detector directly
"""
import os
import sys
import argparse
from deepfake_detector import test_detection

def main():
    parser = argparse.ArgumentParser(description='Run the BNext-L deepfake detector')
    parser.add_argument('image_path', help='Path to the image to analyze')
    args = parser.parse_args()
    
    image_path = args.image_path
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    # Check if the model exists
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              'models', 'bnext_l_pretrained.pth.tar')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please download the BNext-L model and place it in the models directory")
        sys.exit(1)
    
    print(f"Analyzing image: {image_path}")
    result = test_detection(image_path)
    
    # Print the results
    print("\n" + "=" * 60)
    print("DEEPFAKE DETECTION RESULTS")
    print("=" * 60)
    
    print(f"\nImage: {image_path}")
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

if __name__ == "__main__":
    main()