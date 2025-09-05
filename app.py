import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="MH Early Signs Detector", page_icon="🧠", layout="centered")
st.title("🧠 Mental Health Early Signs Detector")
st.caption("Educational demo — not medical advice.")

# -----------------------------
# Load Model Once
# -----------------------------
@st.cache_resource
def load_model():
    model_id = "hugps/mh-bert"  # replace with your fine-tuned model if available
    fallback_model = "distilbert-base-uncased-finetuned-sst-2-english"

    try:
        return pipeline("text-classification", model=model_id, return_all_scores=True)
    except Exception:
        # silent fallback (no message shown)
        return pipeline("text-classification", model=fallback_model, return_all_scores=True)

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
            results = clf(t)[0]

            # Extract p1 (score for POSITIVE or LABEL_1)
            p1 = 0.0
            for r in results:
                if r["label"] in ["LABEL_1", "POSITIVE"]:
                    p1 = r["score"]

            # Decision
            if p1 >= thr:
                st.error(f"⚠️ Potential MH sign — p1={p1:.2f}, thr={thr:.2f}")
                st.info("If this is about you, consider reaching out to someone you trust or a professional. ❤️")
            else:
                st.success(f"✅ Non-issue — p1={p1:.2f}, thr={thr:.2f}")

        with st.expander("Details"):
            st.json(results)

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
        results = clf(s)[0]
        p1 = 0.0
        for r in results:
            if r["label"] in ["LABEL_1", "POSITIVE"]:
                p1 = r["score"]
        label = "Potential MH sign" if p1 >= thr else "Non-issue"
        st.write(f"**{label} (p1={p1:.2f})** — {s}")
