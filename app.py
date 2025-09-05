import streamlit as st
from transformers import pipeline

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="MH Early Signs Detector", page_icon="🧠", layout="centered")
st.title("🧠 Mental Health Early Signs Detector")
st.caption("Educational demo — not medical advice.")

# -----------------------------
# Load Model Once (with fallback)
# -----------------------------
@st.cache_resource
def load_model():
    user_model_id = "hugps/mh-bert"  # ⚠️ Replace with your actual model ID if it exists
    fallback_model_id = "distilbert-base-uncased-finetuned-sst-2-english"

    try:
        return pipeline("text-classification", model=user_model_id)
    except Exception as e:
        st.warning(f"Could not load model '{user_model_id}'. Falling back to public model. Error: {e}")
        return pipeline("text-classification", model=fallback_model_id)

clf = load_model()

# -----------------------------
# UI
# -----------------------------
text = st.text_area("Enter text", height=160, placeholder="Type or paste text…")
thr = st.slider("Alert threshold (class 1)", 0.50, 0.90, value=0.65, step=0.01)

if st.button("Analyze"):
    t = (text or "").strip()
    if len(t) < 3:
        st.warning("Please enter a longer text.")
    else:
        with st.spinner("Analyzing..."):
            out = clf(t)[0]
            label, score = out["label"], out["score"]

            # Map labels
            mapping = {"LABEL_0": "Non-issue", "LABEL_1": "Potential MH sign"}
            friendly = mapping.get(label, label)

            st.write("### Result")
            if friendly == "Potential MH sign" and score >= thr:
                st.error(f"⚠️ {friendly} — p1={score:.2f}, thr={thr:.2f}")
                st.info("If this is about you, consider reaching out to someone you trust or a professional. ❤️")
            else:
                st.success(f"✅ Non-issue — p1={score:.2f}, thr={thr:.2f}")

        with st.expander("Details"):
            st.json(out)

# -----------------------------
# Examples
# -----------------------------
st.write("---")
if st.button("Try examples"):
    examples = [
        "I had a relaxing day at the park with my family.",
        "Lately I feel empty and it’s hard to get out of bed."
    ]
    for s in examples:
        out = clf(s)[0]
        label, score = out["label"], out["score"]
        mapping = {"LABEL_0": "Non-issue", "LABEL_1": "Potential MH sign"}
        friendly = mapping.get(label, label)
        st.write(f"**{friendly} ({score:.2f})** — {s}")
