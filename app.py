# app.py
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1) Which model to load (your Hugging Face repo)
MODEL_ID = "abc/mh-bert"  # <-- your model

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def predict(text):
    # Turn text into model inputs
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    # Get model output (no training here)
    with torch.no_grad():
        out = model(**enc)
        probs = F.softmax(out.logits, dim=-1).cpu().numpy()[0]  # [prob_class0, prob_class1]
    return probs

# ----------- Streamlit UI -----------
st.set_page_config(page_title="Mental Health Early Signs Detector", page_icon="🧠", layout="centered")
st.title("🧠 Mental Health Early Signs Detector")
st.caption("Demo only — not medical advice.")

text = st.text_area("Paste a post/comment to analyze:", height=140, placeholder="I feel ...")

if st.button("Analyze"):
    if text.strip():
        probs = predict(text)
        pred_class = int(probs.argmax())
        confidence = float(probs[pred_class])
        # Use nice labels if saved in your model config; else fallback
        labels = model.config.id2label if model.config.id2label else {0: "Non-issue", 1: "Potential MH sign"}
        st.write(f"**Prediction:** {labels.get(pred_class, pred_class)}")
        st.write(f"**Confidence:** {confidence:.2%}")
    else:
        st.warning("Please type something first 🙂")
