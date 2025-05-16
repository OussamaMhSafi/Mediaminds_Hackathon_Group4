# app.py
import sys
import os

def create_app():
    import streamlit as st
    import requests
    from PIL import Image
    from io import BytesIO
    import mimetypes
    import logging
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("deepfake_verifier")
    
    def load_graph():
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
            
        from studio.image_classification import create_image_classification_graph
        return create_image_classification_graph()
    
    st.set_page_config(page_title="Deepfake Image Verifier", layout="centered")
    st.title("🧠 Deepfake Image Verifier")
    st.markdown("Paste an image URL to verify if it's real or fake using AI.")
    
    @st.cache_resource
    def get_cached_graph():
        return load_graph()
    
    def is_valid_image_url(url):
        """More robust image URL validation function"""
        try:
            # Custom headers to mimic a browser (many image hosts check this)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            # Make request with headers and allow redirects
            response = requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=10)
            response.raise_for_status()
            
            # Check Content-Type header
            content_type = response.headers.get('content-type', '')
            logger.info(f"URL: {url}, Content-Type: {content_type}")
            
            # Primary check - content-type from headers
            if content_type and content_type.startswith('image/'):
                return True, response
                
            # Fallback check - try to infer from URL if header is empty
            if not content_type:
                # Guess content type from URL extension
                url_path = url.split('?')[0]  # Remove query params
                guessed_type = mimetypes.guess_type(url_path)[0]
                logger.info(f"Guessed Content-Type from URL: {guessed_type}")
                
                if guessed_type and guessed_type.startswith('image/'):
                    return True, response
                    
            # Final fallback - try to open as image
            try:
                Image.open(BytesIO(response.content))
                logger.info("Successfully opened as image despite content-type")
                return True, response
            except Exception as e:
                logger.error(f"Failed to open as image: {str(e)}")
                return False, None
                
            return False, None
            
        except Exception as e:
            logger.error(f"Error validating image URL: {str(e)}")
            return False, None
    
    image_url = st.text_input("Enter image URL:", placeholder="https://example.com/image.jpg")

    if image_url:
        is_valid, response = is_valid_image_url(image_url)
        
        if not is_valid:
            st.error(f"The URL does not appear to point to a valid image. Please check the URL and try again.")
        else:
            try:
                img = Image.open(BytesIO(response.content))
                st.image(img, caption="Analyzed Image", use_column_width=True)

                if st.button("Analyze Image"):
                    graph = get_cached_graph()
                    
                    with st.spinner("Analyzing... This may take a few moments"):
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

                        progress_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        
                        steps = ["Loading image", "Describing image", "Searching for information", 
                                "Performing reverse image search", "Analyzing results"]
                        
                        for i, step in enumerate(steps):
                            progress_placeholder.text(f"Step {i+1}/{len(steps)}: {step}")
                            progress_bar.progress((i+1)/len(steps))
                        
                        result = graph.invoke(initial_state)
                        
                        progress_placeholder.empty()
                        progress_bar.empty()
                        
                        decision = result.get("decision", {})
                        
                        if decision:
                            classification = decision.get("classification", "UNKNOWN")
                            confidence = decision.get("confidence", 0)
                            
                            st.subheader("🔍 Verdict")
                            
                            if classification == "REAL":
                                st.success(f"Classification: REAL (Confidence: {confidence}%)")
                            elif classification == "FAKE":
                                st.error(f"Classification: FAKE (Confidence: {confidence}%)")
                            else:
                                st.warning(f"Classification: {classification} (Confidence: {confidence}%)")
                            
                            st.subheader("📘 Explanation")
                            explanation = decision.get("explanation", "No explanation available.")
                            st.markdown(explanation)
                            
                            st.subheader("🔗 Sources")
                            sources = decision.get("sources", [])
                            if sources:
                                for src in sources:
                                    st.markdown(f"- [{src}]({src})")
                            else:
                                st.markdown("No sources found.")
                                
                            with st.expander("Show Analysis Details"):
                                st.markdown("### Image Description")
                                st.write(result.get("description", "No description available."))
                                
                                st.markdown("### Search Query Used")
                                st.write(result.get("search_query", "No search query available."))
                                
                                st.markdown("### Similar Images Found")
                                st.write(f"Found {result.get('similar_images_count', 0)} visually similar images.")
                        else:
                            st.error("Failed to classify the image. The model did not return a decision.")
            except Exception as e:
                st.error(f"Failed to process the image: {e}")

    st.markdown("---")
    st.markdown("Powered by LangChain and Streamlit")

if __name__ == "__main__":
    if not 'streamlit' in sys.modules:
        print("WARNING: This script should be run using 'streamlit run app.py'")
    
    create_app()