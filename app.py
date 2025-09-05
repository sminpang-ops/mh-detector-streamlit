# app.py
# -------------------------------
# Streamlit demo for mental-health early sign detection
# via Hugging Face Inference API (no local TF/PyTorch).
# -------------------------------

import time
import json
import requests
import streamlit as st

# ---------- Settings (read from Streamlit Secrets if available) ----------
MODEL_ID = st.secrets.get("MODEL_ID", "hugps/mh-bert-pt")  # try your PyTorch repo first
HF_TOKEN = st.secrets.get("HF_TOKEN", "")                  # optional but recommended
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# ---------- Helpers ----------
@st.cache_data(ttl=1800, show_spinner=False)
def wait_until_ready(model_id: str, max_wait_s: int = 240) -> bool:
    """
    Poll /models/{id} until the model is loaded or max_wait_s elapses.
    Returns True if the model looks ready (HTTP 200 and not 'loading').
    """
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                # When loading, the body often contains {"state":"loading"}
                try:
                    info = r.json()
                    state = (info.get("state") or "").lower()
                except Exception:
                    state = ""
                if state != "loading":
                    return True
        except requests.RequestException:
            pass
        time.sleep(3)
    return False


def classify(text: str, tries: int = 8, timeout: int = 180):
    """
    Call the Inference API with retries & generous timeouts.
    Returns a list of {label, score} dicts (top-level list).
    """
    url = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    payload = {
        "inputs": text,
        "parameters": {"return_all_scores": True},
        "options": {"wait_for_model": True, "use_cache": True},
    }

    for i in range(tries):
        try:
            r = requests.post(url, headers=HEADERS, json=payload, timeout=timeout)
            if r.status_code == 503:
                # model is still loading; backoff and retry
                time.sleep(4 * (i + 1))
                continue
            if r.status_code == 404:
                raise RuntimeError(
                    f"Model not found at {MODEL_ID}. Check the repo path and visibility."
                )
            r.raise_for_status()

            try:
                data = r.json()
            except Exception:
                snippet = (r.text or "")[:200]
                raise RuntimeError(f"Inference returned non-JSON: {snippet}")

            # API may return [[{...},{...}]] or [{...},{...}]
            if isinstance(data, list) and data and isinstance(data[0], list):
                data = data[0]
            return data

        except requests.exceptions.Timeout:
            time.sleep(4 * (i + 1))
        except requests.exceptions.RequestException:
            time.sleep(4 * (i + 1))

    raise RuntimeError("Inference did not complete in time. Please try again later.")


def friendly(scores):
    """
    Map generic LABEL_0/1 to human-friendly names.
    """
    out = {}
    for d in scores:
        lab = d.get("label", "")
        pretty = {"LABEL_0": "Non-issue", "LABEL_1": "Potential MH sign"}.get(lab, lab)
        out[pretty] = float(d.get("score", 0.0))
    # Ensure keys exist
    out.setdefault("Non-issue", 0.0)
    out.setdefault("Potential MH sign", 0.0)
    return out


# ---------- UI ----------
st.set_page_config(page_title="Mental Health Early Signs Detector", page_icon="🧠")
st.title("🧠 Mental Health Early Signs Detector")
st.caption("Demo — not medical advice.")

# Warm up the model once per session
with st.spinner("Warming up model (first call can take up to a minute)…"):
    _ready = wait_until_ready(MODEL_ID, max_wait_s=240)

text = st.text_area(
    "Enter text",
    height=140,
    placeholder="Type or paste text… (e.g., “Lately I feel empty and it’s hard to get out of bed.”)"
)
thr = st.slider("Alert threshold (class 1)", 0.50, 0.90, 0.65, 0.01)

if st.button("Analyze"):
    t = (text or "").strip()
    if len(t) < 3:
        st.warning("Please enter a longer text.")
    else:
        try:
            raw = classify(t)
            scores = friendly(raw)
            p1 = scores["Potential MH sign"]
            label = "Potential MH sign" if p1 >= thr else "Non-issue"

            if label == "Potential MH sign":
                st.error(f"⚠️ {label} — p1={p1:.2f} (threshold={thr:.2f})")
            else:
                st.success(f"✅ {label} — p1={p1:.2f} (threshold={thr:.2f})")

            with st.expander("See raw scores"):
                st.json(raw)
        except Exception as e:
            st.error(f"API error: {e}")

st.write("---")
if st.button("Try examples"):
    examples = [
        "I had a relaxing day at the park with my family.",
        "Lately I feel empty and it’s hard to get out of bed.",
    ]
    for s in examples:
        try:
            raw = classify(s)
            scores = friendly(raw)
            p1 = scores["Potential MH sign"]
            label = "Potential MH sign" if p1 >= thr else "Non-issue"
            st.write(f"**{label}** — p1={p1:.2f} — _{s}_")
        except Exception as e:
            st.write(f"Error on example: {e}")
