import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

MODEL_ID = "hugps/mh-bert"   # your HF repo

@st.cache_resource(show_spinner=True)
def load_model():
    token = st.secrets.get("HF_TOKEN", None)  # only needed if the HF repo is private
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    mdl = TFAutoModelForSequenceClassification.from_pretrained(MODEL_ID, token=token)
    mdl.trainable = False
    # Friendly labels
    if getattr(mdl.config, "id2label", None) in (None, {0: "LABEL_0", 1: "LABEL_1"}):
        mdl.config.id2label = {0: "Non-issue", 1: "Potential MH sign"}
        mdl.config.label2id = {"Non-issue": 0, "Potential MH sign": 1}
    return tok, mdl

tokenizer, model = load_model()

st.set_page_config(page_title="MH Early Signs Detector", page_icon="🧠", layout="centered")
st.title("🧠 Mental Health Early Signs Detector")
st.caption("Educational demo — not medical advice.")

text = st.text_area("Enter text", height=160, placeholder="Type or paste text…")
thr  = st.slider("Alert threshold (class 1)", 0.50, 0.90, value=0.65, step=0.01)

if st.button("Analyze"):
    if len((text or "").strip()) < 3:
        st.warning("Please enter a longer text.")
    else:
        enc = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
        out = model(enc)
        probs = tf.nn.softmax(out.logits, axis=-1).numpy()[0]  # [p0, p1]
        p0, p1 = float(probs[0]), float(probs[1])
        label = "Potential MH sign" if p1 >= thr else "Non-issue"

        st.write("### Result")
        if label == "Potential MH sign":
            st.error(f"⚠️ {label} — p1={p1:.2f}, p0={p0:.2f}, thr={thr:.2f}")
            st.info("If this is about you, consider reaching out to someone you trust or a professional. ❤️")
        else:
            st.success(f"✅ {label} — p1={p1:.2f}, p0={p0:.2f}, thr={thr:.2f})")

st.write("---")
if st.button("Try example sentences"):
    for s in [
        "I had a relaxing day at the park with my family.",
        "Lately I feel empty and it’s hard to get out of bed."
    ]:
        enc = tokenizer(s, return_tensors="tf", truncation=True, padding=True, max_length=128)
        out = model(enc)
        probs = tf.nn.softmax(out.logits, axis=-1).numpy()[0]
        p0, p1 = float(probs[0]), float(probs[1])
        lbl = "Potential MH sign" if p1 >= thr else "Non-issue"
        st.write(f"**{lbl}** — p1={p1:.2f} · _{s}_")
