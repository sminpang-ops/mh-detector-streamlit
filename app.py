# app.py — Streamlit + Hugging Face Inference API (no TF/torch)

import time
import json
import requests
import streamlit as st

# ======== CONFIG ========
# TEMP quick-test model to prove everything works:
# MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"

# When you’re ready to use your model, set:
MODEL_ID = "hugps/mh-bert"   # or "hugps/mh-bert-pt" after conversion (see Step 6)

API_URL  = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# Put your token in Streamlit Secrets (Manage app → Settings → Secrets)
HF_TOKEN = st.secrets.get("HF_TOKEN", None)
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# ======== HELPERS ========
def ping_model(model_id: str):
    """Show whether the model repo is reachable (helps debug 404/401)."""
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        st.caption(f"Ping /api/models/{model_id} → {r.status_code}")
        if r.text:
            st.code(r.text[:300], language="json")
    except Exception as e:
        st.caption(f"Ping failed: {e}")

def classify_remote(text: str):
    """
    Call HF Inference API directly.
    - Ask server to warm the model (wait_for_model).
    - Handle cold starts/timeouts with retries.
    Returns: list of {label, score}.
    """
    payload = {
        "inputs": text,
        "parameters": {"return_all_scores": True},
        "options": {"wait_for_model": True}
    }

    max_tries = 6
    base_sleep = 4  # seconds

    for i in range(max_tries):
        try:
            r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=240)
            if r.status_code == 503:
                wait = base_sleep * (i + 1)
                st.info(f"Model is warming up (503). Retrying in {wait}s…")
                time.sleep(wait)
                continue

            if r.status_code == 404:
                raise RuntimeError(
                    f"Model not found at {MODEL_ID}. "
                    f"Open https://huggingface.co/{MODEL_ID} to confirm the path."
                )

            r.raise_for_status()
            data = r.json()
            # Normalize shape: [[{label, score}, ...]] -> [{...}, ...]
            if isinstance(data, list) and data and isinstance(data[0], list):
                data = data[0]
            return [{"label": d["label"], "score": float(d["score"])} for d in data]

        except requests.exceptions.ReadTimeout:
            wait = base_sleep * (i + 1)
            st.info(f"Inference timed out. Retrying in {wait}s…")
            time.sleep(wait)
        except requests.exceptions.RequestException as e:
            wait = base_sleep * (i + 1)
            st.info(f"Network error: {e}. Retrying in {wait}s…")
            time.sleep(wait)

    raise RuntimeError("Inference did not complete in time. Please try again shortly.")

def friendly_scores(raw):
    mapping = {"LABEL_0": "Non-issue", "LABEL_1": "Potential MH sign"}
    return {mapping.get(d["label"], d["label"]): d["score"] for d in raw}

# (Optional) warmup once at startup
@st.cache_resource
def _warmup():
    try:
        _ = classify_remote("hello")
    except Exception:
        pass
    return True

# ======== UI ========
st.set_page_config(page_title="MH Early Signs Detector", page_icon="🧠", layout="centered")
st.title("🧠 Mental Health Early Signs Detector")
st.caption("Educational demo — not medical advice.")

# Show a quick ping so you can see 200 / 404, etc.
ping_model(MODEL_ID)

text = st.text_area("Enter text", height=160, placeholder="Type or paste text…")
thr  = st.slider("Alert threshold (class 1)", 0.50, 0.90, value=0.65, step=0.01)

if st.button("Analyze"):
    t = (text or "").strip()
    if len(t) < 3:
        st.warning("Please enter a longer text.")
    else:
        try:
            _ = _warmup()
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
