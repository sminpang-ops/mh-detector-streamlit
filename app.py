import json
import streamlit as st
from huggingface_hub import InferenceClient
import requests

MODEL_ID = "hugps/mh-bert"  # <-- your model repo
HF_TOKEN = st.secrets.get("HF_TOKEN", None)

@st.cache_resource(show_spinner=True)
def get_client():
    # If token is None, requests may be rate-limited or rejected; token is recommended
    return InferenceClient(model=MODEL_ID, token=HF_TOKEN)

client = get_client()

st.set_page_config(page_title="MH Early Signs Detector", page_icon="🧠", layout="centered")
st.title("🧠 Mental Health Early Signs Detector")
st.caption("Educational demo — not medical advice.")

text = st.text_area("Enter text", height=160, placeholder="Type or paste text…")
thr  = st.slider("Alert threshold (class 1)", 0.50, 0.90, value=0.65, step=0.01)

def classify_remote(txt: str):
    """
    Uses HF Inference API via the official client. Returns list of {label, score}.
    """
    out = client.text_classification(txt, wait_for_model=True, details=True)
    cleaned = []
    for o in out:
        # o can be an object or dict depending on hub version
        label = getattr(o, "label", o.get("label"))
        score = float(getattr(o, "score", o.get("score")))
        cleaned.append({"label": label, "score": score})
    return cleaned

def map_scores(raw):
    # Map labels to friendly names if needed
    mapping = {"LABEL_0": "Non-issue", "LABEL_1": "Potential MH sign"}
    return {mapping.get(d["label"], d["label"]): d["score"] for d in raw}

if st.button("Analyze"):
    t = (text or "").strip()
    if len(t) < 3:
        st.warning("Please enter a longer text.")
    else:
        try:
            raw = classify_remote(t)
            scores = map_scores(raw)
            p1 = float(scores.get("Potential MH sign", 0.0))
            label = "Potential MH sign" if p1 >= thr else "Non-issue"

            st.write("### Result")
            if label == "Potential MH sign":
                st.error(f"⚠️ {label} — p1={p1:.2f}, thr={thr:.2f}")
                st.info("If this is about you, consider reaching out to someone you trust or a professional. ❤️")
            else:
                st.success(f"✅ {label} — p1={p1:.2f}, thr={thr:.2f}")

            with st.expander("Details"):
                st.json(raw)
        except Exception as e:
            # Show exact response to debug auth / 401, etc.
            st.error(f"API error: {e}\nIf your model is private or rate-limited, set HF_TOKEN in Streamlit Secrets.")
            # Optional: show model metadata ping
            headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
            r = requests.get(f"https://huggingface.co/api/models/{MODEL_ID}", headers=headers, timeout=30)
            st.code(f"Ping /api/models → {r.status_code}\n{r.text[:400]}", language="text")

st.write("---")
if st.button("Try examples"):
    for s in [
        "I had a relaxing day at the park with my family.",
        "Lately I feel empty and it’s hard to get out of bed."
    ]:
        raw = classify_remote(s)
        scores = map_scores(raw)
        p1 = float(scores.get("Potential MH sign", 0.0))
        st.write(f"**p1={p1:.2f}** — _{s}_")
