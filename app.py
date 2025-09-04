import time
import json
import requests
import streamlit as st

MODEL_ID = "hugps/mh-bert"
API_URL  = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HF_TOKEN = st.secrets.get("HF_TOKEN")  # recommended even for public models
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def classify_remote(text: str):
    """
    Call HF Inference API directly.
    - Handles cold start (503) by retrying.
    - Returns a list [{"label": "...", "score": ...}, ...].
    """
    payload = {
        "inputs": text,
        "parameters": {"return_all_scores": True},
        "options": {"wait_for_model": True}   # <-- this replaces the old kwarg
    }

    for attempt in range(2):  # one retry if cold
        r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        if r.status_code == 503:
            time.sleep(5)
            continue
        r.raise_for_status()
        data = r.json()

        # Normalize shapes:
        # - with return_all_scores=True you usually get [[{label, score}, ...]]
        # - sometimes you may get just [{label, score}, ...]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            data = data[0]
        return [{"label": d["label"], "score": float(d["score"])} for d in data]

    raise RuntimeError("Model still loading. Try again shortly.")

def map_scores(raw):
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
            else:
                st.success(f"✅ {label} — p1={p1:.2f}, thr={thr:.2f}")

            with st.expander("Details"):
                st.json(raw)
        except Exception as e:
            st.error(f"API error: {e}")
            # Optional quick ping for debugging
            r = requests.get(f"https://huggingface.co/api/models/{MODEL_ID}",
                             headers=HEADERS, timeout=30)
            st.code(f"Ping /api/models → {r.status_code}\n{r.text[:400]}")
