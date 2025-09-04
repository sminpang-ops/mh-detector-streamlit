import os
import json
import time
import requests
import streamlit as st

MODEL_ID = "hugps/mh-bert"        # your model repo on HF
API_URL  = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# Put your token in Streamlit Secrets (Manage app → Settings → Secrets)
HF_TOKEN = st.secrets.get("HF_TOKEN", None)
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

st.set_page_config(page_title="MH Early Signs Detector", page_icon="🧠", layout="centered")
st.title("🧠 Mental Health Early Signs Detector")
st.caption("Educational demo — not medical advice.")

text = st.text_area("Enter text", height=160, placeholder="Type or paste text…")
thr  = st.slider("Alert threshold (class 1)", 0.50, 0.90, value=0.65, step=0.01)

def classify_remote(txt: str):
    # HF text-classification expects {"inputs": "..."}
    # Retry once if the model is sleeping ("loading" status)
    payload = {"inputs": txt, "parameters": {"return_all_scores": True}}
    for attempt in range(2):
        r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        if r.status_code == 503 and "loading" in r.text.lower():
            time.sleep(5)  # warmup
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Model still loading. Try again in a moment.")

if st.button("Analyze"):
    t = (text or "").strip()
    if len(t) < 3:
        st.warning("Please enter a longer text.")
    else:
        try:
            out = classify_remote(t)
            # Expected format: [[{"label":"LABEL_0","score":..},{"label":"LABEL_1","score":..}]]
            scores = out[0]
            # map to friendly labels
            mapping = {"LABEL_0": "Non-issue", "LABEL_1": "Potential MH sign"}
            sdict = {mapping.get(item["label"], item["label"]): float(item["score"]) for item in scores}
            p1 = sdict.get("Potential MH sign", 0.0)
            label = "Potential MH sign" if p1 >= thr else "Non-issue"

            st.write("### Result")
            if label == "Potential MH sign":
                st.error(f"⚠️ {label} — p1={p1:.2f}, thr={thr:.2f}")
                st.info("If this is about you, consider reaching out to someone you trust or a professional. ❤️")
            else:
                st.success(f"✅ {label} — p1={p1:.2f}, thr={thr:.2f}")

            with st.expander("Raw API response"):
                st.code(json.dumps(out, indent=2))
        except Exception as e:
            st.error(f"API error: {e}\n\nIf your model is private, add HF_TOKEN in Streamlit Secrets.")
