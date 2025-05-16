import streamlit as st
from studio.image_classification import create_image_classification_graph
from PIL import Image
import requests
from io import BytesIO
import json

# Initialize the graph
graph = create_image_classification_graph()

st.set_page_config(page_title="Deepfake Image Verifier", layout="centered")
st.title("🧠 Deepfake Image Verifier")
st.markdown("Paste an image URL to verify if it's real or fake using AI.")

image_url = st.text_input("Enter image URL:", placeholder="https://example.com/image.jpg")

if image_url:
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        st.image(img, caption="Analyzed Image", use_column_width=True)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                initial_state = {
                    "image": image_url,
                    "image_data": {},
                    "description": "",
                    "web_scrape_results": None,
                    "search_query": "",
                    "classification": "",
                    "sources": [],
                    "decision": None,
                    "similar_images_count": 0,
                    "visual_match_urls": [],
                    "visual_match_contents": []
                }

                result = graph.invoke(initial_state)
                decision = result.get("decision", {})

                st.subheader("🔍 Verdict")
                st.markdown(f"**Classification:** `{decision.get('classification')}`")
                st.markdown(f"**Confidence:** `{decision.get('confidence')}%`")

                st.subheader("📘 Explanation")
                st.write(decision.get("explanation", "No explanation available."))

                st.subheader("🔗 Sources")
                for src in decision.get("sources", []):
                    st.markdown(f"- [{src}]({src})")

    except Exception as e:
        st.error(f"Failed to process the image: {e}")
