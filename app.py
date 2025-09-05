# app.py
import os
import time
import json
import requests
import streamlit as st

# ------------------ Config ------------------
# Put these in Streamlit Secrets (Settings → Secrets):
# MODEL_ID="hugps/mh-bert-pt"
# HF_TOKEN="hf_xxx"           # only needed if the repo is private/rate-limited

MODEL_ID = st.secrets.get("MODEL_ID", "hugps/mh-bert-pt")
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

LABEL_MAP = {"LABEL_0": "Non-issue", "LABEL_1": "Potential MH sign"}  # fallback pretty names

# ------------------ Inference ------------------
def classify(text: str, tries: int = 5, timeout: int = 90):
    """
    Call the HF Inference API with retries.
    Returns: list[ {label: str, score: float}, ... ] for the best candidate set.
    Raises RuntimeError on persistent failure.
    """
    url = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    payload = {
        "inputs": text,
        "parameters": {"return_all_scores": True},
        "options": {"wait_for_model": True},
    }

    for i in range(tries):
        try:
            r = requests.post(url, headers=HEADERS, json=payload, timeout=timeout)
            # 503: model still spinning up — wait and retry
            if r.status_code == 503:
                time.sleep(3 * (i + 1))
                continue
            # 404: wrong model path
            if r.status_code == 404:
                raise RuntimeError(
                    f"Model not found at {MODEL_ID}. Check the repo path or Secrets."
                )
            r.raise_for_status()

            # parse JSON safely
            try:
                data = r.json()
            except Exception:
                snippet = (r.text or "")[:200]
                raise RuntimeError(f"Inference returned non-JSON: {snippet}")

            # Normalize shape: [[{label,score},...]] -> [{label,score},...]
            if isinstance(data, list) and data and isinstance(data[0], list):
                data = data[0]
            return data

        except requests.exceptions.Timeout:
            time.sleep(3 * (i + 1))
        except requests.exceptions.RequestException:
            time.sleep(3 * (i + 1))

    raise RuntimeError("Inference did not complete in time. Please try again.")

def prettify(scores):
    """
    Convert LABEL_0/LABEL_1 to readable names if necessary.
    """
    out = []
    for d in scores:
        lab = d.get("label", "")
        pretty = LABEL_MAP.get(lab, lab)
        out.append({"label": pretty, "score": float(d.get("score", 0.0))})
    return out

# ------------------ UI ------------------
st.set_page_config(page_title="MH Detector", page_icon="🧠")
st.title("🧠 Mental Health Early Signs Detector")
st.caption("Demo — not medical advice.")

text = st.text_area("Enter text", height=140, placeholder="Type or paste text…")
thr  = st.slider("Alert threshold (class 1)", 0.50, 0.90, 0.65, 0.01)

col1, col2 = st.columns([1,1])
with col1:
    run = st.button("Analyze", type="primary")
with col2:
    examples = st.button("Try examples")

if run:
    t = (text or "").strip()
    if len(t) < 3:
        st.warning("Please enter a longer text.")
    else:
        with st.spinner("Analyzing…"):
            try:
                raw = classify(t)
                scores = prettify(raw)

                # extract class-1 prob
                p1 = 0.0
                for s in scores:
                    if s["label"].lower().startswith("potential"):
                        p1 = s["score"]
                        break

                label = "Potential MH sign" if p1 >= thr else "Non-issue"
                if label == "Potential MH sign":
                    st.error(f"⚠️ {label} — p1={p1:.2f}, thr={thr:.2f}")
                else:
                    st.success(f"✅ {label} — p1={p1:.2f}, thr={thr:.2f}")

                with st.expander("Details"):
                    st.json(scores)

            except Exception as e:
                st.error(f"API error: {e}")

if examples:
    demo_texts = [
        "I had a relaxing day at the park with my family.",
        "Lately I feel empty and it’s hard to get out of bed.",
    ]
    for s in demo_texts:
        try:
            raw = classify(s)
            scores = prettify(raw)
            p1 = 0.0
            for d in scores:
                if d["label"].lower().startswith("potential"):
                    p1 = d["score"]
                    break
            verdict = "Potential MH sign" if p1 >= thr else "Non-issue"
            st.write(f"**{verdict}** — p1={p1:.2f} — _{s}_")
        except Exception as e:
            st.warning(f"Could not score example. {e}")

st.write("---")
st.caption(
    "Model: "
    f"[{MODEL_ID}](https://huggingface.co/{MODEL_ID}) · "
    "Powered by Hugging Face Inference API."
)
