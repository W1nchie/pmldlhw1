"""
Streamlit Web Application for Handwritten Digit Recognition
"""

import streamlit as st
import requests
import numpy as np
from PIL import Image, ImageOps
import io

st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)

API_URL = "http://api:8000"


def to_8x8_vector(image: Image.Image) -> np.ndarray:
    gray = image.convert("L")
    gray = ImageOps.invert(gray)
    gray = gray.resize((8, 8))
    arr = np.array(gray, dtype=np.float32)
    arr = (arr / 255.0) * 16.0
    return arr.reshape(1, -1)


def main():
    st.title("🔢 Handwritten Digit Recognizer")
    st.markdown("Upload a digit image or draw one to get a prediction.")

    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("API is running and healthy")
        else:
            st.error("API is not responding properly")
            return
    except requests.exceptions.RequestException:
        st.error("Cannot connect to API. Make sure the API container is running.")
        return

    st.header("Upload Image")
    uploaded_file = st.file_uploader("Upload a digit image (PNG/JPG)", type=["png", "jpg", "jpeg"])

    vector = None
    preview = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        preview = image.copy()
        vector = to_8x8_vector(image)

    if st.button("Predict Digit", use_container_width=True):
        if vector is None:
            st.warning("Please upload an image first.")
        else:
            st.image(preview, caption="Uploaded Image", width=200)
            payload = {"pixels": vector.flatten().tolist()}
            with st.spinner("Making prediction..."):
                try:
                    res = requests.post(f"{API_URL}/predict_pixels", json=payload, timeout=10)
                    if res.status_code == 200:
                        result = res.json()
                        st.success("Prediction completed!")
                        st.metric("Predicted Digit", result["prediction"])
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                    else:
                        st.error(f"API Error: {res.status_code}")
                        st.error(res.text)
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection Error: {str(e)}")

    with st.expander("About this App"):
        st.markdown(
            """
            This app recognizes handwritten digits using a Support Vector Classifier
            trained on scikit-learn's digits dataset (8x8 images). Upload an image and
            the app will resize and normalize it to the expected format.
            """
        )


if __name__ == "__main__":
    main()
