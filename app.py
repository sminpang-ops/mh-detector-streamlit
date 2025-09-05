import streamlit as st
from transformers import pipeline

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Mental Health Early Signs Detector", page_icon="🧠")
st.title("🧠 Mental Health Early Signs Detector")
st.caption("Demo — not medical advice.")

# -----------------------------
# Load Model Once (Cached)
# -----------------------------
@st.cache_resource
def load_model():
    # Replace with your fine-tuned model ID or local path
    model_id = "hugps/mh-bert"   # e.g., "your-username/your-model"
    return pipeline("text-classification", model=model_id)

nlp = load_model()

# -----------------------------
# UI
# -----------------------------
user_input = st.text_area("Enter a comment:", placeholder="Type something here...")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter a comment before analyzing.")
    else:
        with st.spinner("Analyzing..."):
            result = nlp(user_input.strip())[0]
            label, score = result["label"], result["score"]

            # Adjust mapping according to your training labels
            if label in ["LABEL_1", "WITH_SIGNS"]:
                st.success(f"⚠️ With early signs (confidence {score:.2f})")
            else:
                st.success(f"✅ Without early signs (confidence {score:.2f})")
