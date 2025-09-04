# app.py
import time
import json
import requests
import streamlit as st

# ---------- Config ----------
MODEL_ID = "hugps/mh-bert"  # your HF model repo
API_URL  = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HF_TOKEN = st.secrets.get("HF_TOKEN")  # optional but recommended
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# ---------- Inference helper ----------
def classify_remote(text: str):
    """
    Call Hugging Face Inference API directly.
    Handles cold start (503) and returns a flat list of {label, score}.
    """
    payload = {
        "inputs": text,
        "parameters": {"return_all_scores": True},
        "options": {"wait_for_model": True},  # warm the model if sleeping
    }

    for _ in range(2):  # retry once if cold
        r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        if r.status_code == 503:
            time.sleep(5)
            continue
        r.raise_for_status()
        data = r.json()
        # Normalize shape: [[{label, score}, ...]] -> [{...}, ...]
        if isinstance(data, list) and data and isinstance(data[0], list):
            data = data[0]
        return [{"label": d["label"], "score": float(d["score"])} for d in data]

    raise RuntimeError("Model still loading. Please try again.")

def friendly_scores(raw):
    mapping = {"LABEL_0": "Non-issue", "LABEL_1": "Potential MH sign"}
    return {mapping.get(d["label"], d["label"]): d["score"] for d in raw}

# ---------- UI ----------
st.set_page_config(page_title="MH Early Signs Detector", page_icon="🧠", layout="centered")
st.title("🧠 Mental Health Early Signs Detector")
st.caption("Educational demo — not medical advice.")

# MAKE SURE THESE ARE DEFINED *BEFORE* you use them:
text = st.text_area("Enter text", height=160, placeholder="Type or paste text…")
thr  = st.slider("Alert threshold (class 1)", 0.50, 0.90, value=0.65, step=0.01)

if st.button("Analyze"):
    t = (text or "").strip()
    if len(t) < 3:
        st.warning("Please enter a longer text.")
    else:
        try:
            raw = classify_remote(t)
            scores = friendly_scores(raw)
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
            st.error(f"API error: {e}")
            # optional debug ping
            ping = requests.get(f"https://huggingface.co/api/models/{MODEL_ID}", headers=HEADERS, timeout=30)
            st.code(f"Ping /api/models → {ping.status_code}\n{ping.text[:400]}", language="text")

st.write("---")
if st.button("Try examples"):
    for s in [
        "I had a relaxing day at the park with my family.",
        "Lately I feel empty and it’s hard to get out of bed."
    ]:
        raw = classify_remote(s)
        scores = friendly_scores(raw)
        p1 = float(scores.get("Potential MH sign", 0.0))
        st.write(f"**p1={p1:.2f}** — _{s}_")
