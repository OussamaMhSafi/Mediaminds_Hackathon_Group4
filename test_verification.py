import os
from image_verification import graph

def test_image(image_path):
    """Test deepfake detection on a single image"""
    if not os.path.exists(image_path):
        print(f"Error: Image path {image_path} does not exist")
        return
    
    print(f"Analyzing image: {image_path}")
    print("Running deepfake detection with BNext-L model...")
    
    # Run the graph with the image path
    result = graph.invoke({"image": image_path})
    
    # Extract the bnext model classification
    image_classes = result.get("image_classes", {})
    image_features = result.get("image_features", {})
    
    # Extract the final decision (which includes web verification)
    decision = result.get("decision", {})
    
    # Print model result
    print("\n" + "=" * 60)
    print("DEEPFAKE DETECTION RESULTS")
    print("=" * 60)
    
    # Print BNext-L model results
    print("\n📊 BNext-L Model Assessment:")
    print(f"  Classification: {image_classes.get('class_name', 'UNKNOWN')}")
    print(f"  Confidence: {image_classes.get('confidence', 0):.2f}")
    if image_classes.get('full_probs'):
        print(f"  Probability Real: {image_classes['full_probs']['real']:.2f}")
        print(f"  Probability Fake: {image_classes['full_probs']['fake']:.2f}")
    
    # Print web verification results
    print("\n🌐 Web Verification Assessment:")
    print(f"  Classification: {decision.get('classification', 'UNKNOWN')}")
    print(f"  Confidence: {decision.get('confidence', 0)}%")
    
    # Print final determination
    print("\n🔍 FINAL DETERMINATION:")
    is_fake = decision.get('classification') == "FAKE"
    confidence = decision.get('confidence', 0)
    
    if is_fake and confidence >= 70:
        print(f"  ❌ HIGH CONFIDENCE FAKE ({confidence}%)")
    elif is_fake and confidence >= 40:
        print(f"  ⚠️ LIKELY FAKE ({confidence}%)")
    elif not is_fake and confidence >= 70:
        print(f"  ✅ HIGH CONFIDENCE REAL ({confidence}%)")
    elif not is_fake and confidence >= 40:
        print(f"  ℹ️ LIKELY REAL ({confidence}%)")
    else:
        print(f"  ❓ UNCERTAIN ({confidence}%)")
    
    # Print explanation
    print("\n📋 Explanation:")
    print(decision.get('explanation', 'No explanation provided'))
    
    # Print sources
    print("\n🔗 Sources:")
    for source in decision.get('sources', []):
        print(f"  - {source}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_verification.py <image_path>")
        sys.exit(1)
    
    test_image(sys.argv[1])