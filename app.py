import streamlit as st
import requests
import os
import json

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Mental Health Early Signs Detector", page_icon="🧠")
st.title("🧠 Mental Health Early Signs Detector")
st.caption("Demo — not medical advice.")

# -----------------------------
# Config
# -----------------------------
MODEL_ID = os.environ.get("MODEL_ID", "hugps/mh-bert")  # change to your HF repo
HF_TOKEN = os.environ.get("HF_TOKEN", "")               # set your Hugging Face token

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# -----------------------------
# Helper Function
# -----------------------------
def analyze_text(text: str):
    """Send text to Hugging Face API and return prediction."""
    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": text},
            timeout=60,
        )
        if response.status_code == 503:
            return "Model is loading... please try again in a moment."
        if response.status_code != 200:
            return f"Error: {response.status_code} {response.text[:200]}"
        
        data = response.json()
        # Assume binary classifier with LABEL_0 = without signs, LABEL_1 = with signs
        if isinstance(data, list) and len(data) > 0:
            pred = data[0]
            label = pred.get("label", "LABEL_0")
            score = pred.get("score", 0.0)
            if label in ["LABEL_1", "WITH_SIGNS"]:
                return f"⚠️ With early signs (confidence {score:.2f})"
            else:
                return f"✅ Without early signs (confidence {score:.2f})"
        return f"Unexpected response: {data}"
    except Exception as e:
        return f"Request failed: {e}"

# -----------------------------
# UI
# -----------------------------
user_input = st.text_area("Enter a comment:", placeholder="Type something here...")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter a comment before analyzing.")
    else:
        with st.spinner("Analyzing..."):
            result = analyze_text(user_input.strip())
        st.success(result)
