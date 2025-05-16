# app.py
import sys
import os

# Only import streamlit-related components inside functions
# to avoid module-level Streamlit calls

def create_app():
    import streamlit as st
    import requests
    from PIL import Image
    from io import BytesIO
    
    # Import the graph creation function when needed
    def load_graph():
        # Add the parent directory to sys.path if needed
        # This helps with importing the studio module
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
            
        from studio.image_classification import create_image_classification_graph
        return create_image_classification_graph()
    
    # Configure the page
    st.set_page_config(page_title="Deepfake Image Verifier", layout="centered")
    st.title("🧠 Deepfake Image Verifier")
    st.markdown("Paste an image URL to verify if it's real or fake using AI.")
    
    # Initialize graph with memoization (only when needed)
    @st.cache_resource
    def get_cached_graph():
        return load_graph()
    
    image_url = st.text_input("Enter image URL:", placeholder="https://example.com/image.jpg")

    if image_url:
        try:
            # Validate the image URL first
            response = requests.get(image_url)
            content_type = response.headers.get('content-type', '')
            
            if not content_type.startswith('image/'):
                st.error(f"URL does not point to an image. Content-Type: {content_type}")
            else:
                img = Image.open(BytesIO(response.content))
                st.image(img, caption="Analyzed Image", use_column_width=True)

                if st.button("Analyze Image"):
                    # Now get the graph (only when needed)
                    graph = get_cached_graph()
                    
                    with st.spinner("Analyzing... This may take a few moments"):
                        # Initialize the state correctly according to ImageClassificationState
                        initial_state = {
                            "image": image_url,
                            "image_data": {},
                            "description": "",
                            "web_scrape_results": None,
                            "search_query": "",
                            "classification": "REAL",
                            "sources": [],
                            "decision": None,
                            "similar_images_count": 0,
                            "visual_match_urls": [],
                            "visual_match_contents": []
                        }

                        # Create a progress display
                        progress_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        
                        # Display the processing steps
                        steps = ["Loading image", "Describing image", "Searching for information", 
                                "Performing reverse image search", "Analyzing results"]
                        
                        for i, step in enumerate(steps):
                            progress_placeholder.text(f"Step {i+1}/{len(steps)}: {step}")
                            progress_bar.progress((i+1)/len(steps))
                        
                        # Invoke the graph with the initial state
                        result = graph.invoke(initial_state)
                        
                        # Clear the progress indicators
                        progress_placeholder.empty()
                        progress_bar.empty()
                        
                        # Extract decision from the result
                        decision = result.get("decision", {})
                        
                        if decision:
                            # Display the classification result with appropriate styling
                            classification = decision.get("classification", "UNKNOWN")
                            confidence = decision.get("confidence", 0)
                            
                            st.subheader("🔍 Verdict")
                            
                            # Use different colors for REAL and FAKE classifications
                            if classification == "REAL":
                                st.success(f"Classification: REAL (Confidence: {confidence}%)")
                            elif classification == "FAKE":
                                st.error(f"Classification: FAKE (Confidence: {confidence}%)")
                            else:
                                st.warning(f"Classification: {classification} (Confidence: {confidence}%)")
                            
                            # Show explanation
                            st.subheader("📘 Explanation")
                            explanation = decision.get("explanation", "No explanation available.")
                            st.markdown(explanation)
                            
                            # Show sources
                            st.subheader("🔗 Sources")
                            sources = decision.get("sources", [])
                            if sources:
                                for src in sources:
                                    st.markdown(f"- [{src}]({src})")
                            else:
                                st.markdown("No sources found.")
                                
                            # Display additional information
                            with st.expander("Show Analysis Details"):
                                st.markdown("### Image Description")
                                st.write(result.get("description", "No description available."))
                                
                                st.markdown("### Search Query Used")
                                st.write(result.get("search_query", "No search query available."))
                                
                                st.markdown("### Similar Images Found")
                                st.write(f"Found {result.get('similar_images_count', 0)} visually similar images.")
                        else:
                            st.error("Failed to classify the image. The model did not return a decision.")
        
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch the image: {e}")
        except Exception as e:
            st.error(f"Failed to process the image: {e}")

    # Add footer
    st.markdown("---")
    st.markdown("Powered by LangChain and Streamlit")

# This ensures no Streamlit functions are called at module level
if __name__ == "__main__":
    # Check if being run via streamlit
    if not 'streamlit' in sys.modules:
        print("WARNING: This script should be run using 'streamlit run app.py'")
        print("Running directly with Python will cause errors.")
        # We could exit here, but let's allow it to continue with a warning
    
    # Call the function containing all Streamlit code
    create_app()