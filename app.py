import os
import json
import time
from typing import Any, Dict, List, Tuple

import requests
import streamlit as st

# -----------------------------
# Page & Theme
# -----------------------------
st.set_page_config(
    page_title="Mental Health Early Signs Detector",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Mental Health Early Signs Detector")
st.caption("Demo — not medical advice.")

# -----------------------------
# Secrets / Env
# -----------------------------
# Preferred: put these in .streamlit/secrets.toml
MODEL_ID = st.secrets.get("MODEL_ID", os.environ.get("MODEL_ID", "hugps/mh-bert"))
HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

INFERENCE_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def ping_model(url: str) -> Tuple[int, str]:
    """
    Ping the Hugging Face Inference Endpoint (or model repo) to warm it up.
    Returns (status_code, preview_text)
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        status = r.status_code
        # Try to parse JSON for a short preview
        try:
            body = r.json()
            preview = json.dumps(body)[:400]
        except Exception:
            preview = (r.text or "")[:400]
        return status, preview
    except requests.exceptions.RequestException as e:
        return 0, f"Ping error: {e}"


def _extract_label_and_score(pred: Any) -> Tuple[str, float]:
    """
    Robustly extract (label, score) from HF inference response.
    Supports common formats:
      - [{"label": "LABEL_0", "score": 0.98}]
      - {"labels": ["POSITIVE", "NEGATIVE"], "scores": [0.8, 0.2]}
      - [[{"label": ..., "score": ...}]] for batch
    """
    if isinstance(pred, list):
        # batch or single
        first = pred[0]
        if isinstance(first, dict) and "label" in first and "score" in first:
            return first["label"], float(first["score"])
        if isinstance(first, list) and len(first) > 0 and isinstance(first[0], dict):
            d = first[0]
            return d.get("label", "LABEL_0"), float(d.get("score", 0.0))

    if isinstance(pred, dict):
        labels = pred.get("labels")
        scores = pred.get("scores")
        if labels and scores and len(labels) == len(scores) and len(labels) > 0:
            # take top-1
            idx = int(max(range(len(scores)), key=lambda i: scores[i]))
            return str(labels[idx]), float(scores[idx])

    # Fallback
    return "UNKNOWN", 0.0


def call_hf_inference(text: str, timeout: int = 60) -> Dict[str, Any]:
    """
    Call Hugging Face Inference API with robust error handling.
    """
    payload = {"inputs": text}
    try:
        r = requests.post(
            INFERENCE_URL,
            headers=HEADERS,
            json=payload,
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "Request to Hugging Face timed out."}
    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": f"Network error: {e}"}

    if r.status_code == 503:
        # Model is loading on HF side
        return {"ok": False, "error": "Model is loading on Hugging Face (503). Try again shortly."}

    if r.status_code != 200:
        # Attach short body for debugging
        short_body = r.text[:400]
        return {"ok": False, "error": f"HF API error {r.status_code}: {short_body}"}

    try:
        pred = r.json()
    except Exception:
        return {"ok": False, "error": "Non-JSON response from HF API."}

    label, score = _extract_label_and_score(pred)
    return {"ok": True, "label": label, "score": score, "raw": pred}


# Optional: local pipeline fallback (CPU). Useful when HF API is slow or private.
# We import lazily so users without transformers don't crash import-time.
@st.cache_resource(show_spinner=False)
def get_local_pipeline():
    """
    Lazy-load a local transformers pipeline, only if the user switches to 'Local CPU' backend.
    """
    try:
        from transformers import pipeline

        # If your fine-tuned model is available locally/by name, use MODEL_ID,
        # otherwise fall back to a sentiment model for demo.
        model_choice = MODEL_ID if MODEL_ID else "distilbert-base-uncased-finetuned-sst-2-english"
        return pipeline("text-classification", model=model_choice)
    except Exception as e:
        return f"Local pipeline failed to load: {e}"


def classify_local(pipe, text: str) -> Dict[str, Any]:
    try:
        out = pipe(text, truncation=True)
        # Most pipelines return a list of {label, score}
        if isinstance(out, list) and out:
            d = out[0]
            return {"ok": True, "label": d.get("label", "UNKNOWN"), "score": float(d.get("score", 0.0)), "raw": out}
        return {"ok": False, "error": f"Unexpected local output: {out}"}
    except Exception as e:
        return {"ok": False, "error": f"Local inference error: {e}"}


# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.subheader("⚙️ Inference Settings")
    backend = st.radio("Backend", ["Hugging Face API", "Local CPU (transformers)"], index=0, help="Use your fine-tuned model on HF or a local CPU pipeline.")
    st.write("**Model ID**:", MODEL_ID or "—")

    if backend == "Hugging Face API":
        status, preview = ping_model(INFERENCE_URL)
        if status == 0:
            st.info("Could not ping HF endpoint (network issue or wrong URL).")
        elif status == 200:
            st.success("HF endpoint reachable ✅")
        else:
            st.warning(f"HF ping status: {status}")
        with st.expander("Ping preview (debug)"):
            st.code(preview or "—", language="json")

    else:
        st.caption("Local pipeline loads on first use and may download model weights.")

# -----------------------------
# Main UI
# -----------------------------
EXAMPLES = [
    "Lately I can't sleep and I feel empty most of the time.",
    "I had a good day today and enjoyed a long walk with friends.",
    "Sometimes I think about giving up, but I'm trying to stay hopeful.",
]

st.write("Enter a short post, comment, or sentence:")
text = st.text_area("Input text", value=EXAMPLES[0], height=140, label_visibility="collapsed")

colA, colB = st.columns([1, 1])
with colA:
    if st.button("🔍 Analyze", use_container_width=True):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            if backend == "Hugging Face API":
                with st.spinner("Warming up / calling Hugging Face…"):
                    result = call_hf_inference(text.strip(), timeout=60)
            else:
                with st.spinner("Loading local pipeline (first run may take a while)…"):
                    pipe = get_local_pipeline()
                    if isinstance(pipe, str):
                        result = {"ok": False, "error": pipe}
                    else:
                        result = classify_local(pipe, text.strip())

            if result.get("ok"):
                st.success("Done ✅")
                label = result.get("label", "UNKNOWN")
                score = result.get("score", 0.0)

                st.markdown(f"**Prediction:** `{label}`")
                st.markdown(f"**Confidence:** `{score:.3f}`")

                with st.expander("See raw model output"):
                    st.code(json.dumps(result.get("raw"), indent=2)[:2000], language="json")
            else:
                st.error(result.get("error", "Unknown error"))

with colB:
    st.write("Examples")
    for eg in EXAMPLES:
        if st.button(f"Use: {eg[:40]}…", key=f"ex_{hash(eg)}", use_container_width=True):
            st.session_state["text"] = eg  # just for UX; not strictly necessary
            st.experimental_rerun()

st.divider()
st.caption(
    "This tool is a research demo and not a diagnostic device. If you or someone you know is struggling, please seek professional help."
)
