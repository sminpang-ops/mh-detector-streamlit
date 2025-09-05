import requests
import streamlit as st

MODEL_ID = "hugps/mh-bert"  # <-- update this if your repo is different
HF_TOKEN = st.secrets.get("HF_TOKEN")
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def ping_model(model_id: str):
    url = f"https://huggingface.co/api/models/{model_id}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    st.write(f"Ping /api/models/{model_id} → {r.status_code}")
    # show some of the body so we can see “not found” or metadata
    st.code(r.text[:400], language="json")

import time
import requests
import streamlit as st

MODEL_ID = "hugps/mh-bert"
API_URL  = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HF_TOKEN = st.secrets.get("HF_TOKEN")
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def classify_remote(text: str):
    """
    Call HF Inference API with:
      - server-side warmup (wait_for_model)
      - exponential backoff on 503/timeouts
      - generous read timeout for cold starts
    Returns list of {label, score}.
    """
    payload = {
        "inputs": text,
        "parameters": {"return_all_scores": True},
        "options": {"wait_for_model": True}  # ask HF to load the model if sleeping
    }

    max_tries = 5
    base_sleep = 4  # seconds

    for i in range(max_tries):
        try:
            # give plenty of time for first response after cold start
            r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=180)
            if r.status_code == 503:
                # model is loading; back off and retry
                wait = base_sleep * (i + 1)
                st.info(f"Model is warming up (503). Retrying in {wait}s…")
                time.sleep(wait)
                continue

            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data and isinstance(data[0], list):
                data = data[0]
            return [{"label": d["label"], "score": float(d["score"])} for d in data]

        except requests.exceptions.ReadTimeout:
            wait = base_sleep * (i + 1)
            st.info(f"Inference timed out. Retrying in {wait}s…")
            time.sleep(wait)
        except requests.exceptions.RequestException as e:
            # Any other network error: brief backoff, then retry
            wait = base_sleep * (i + 1)
            st.info(f"Network error: {e}. Retrying in {wait}s…")
            time.sleep(wait)

    # If we’re here, all tries failed
    raise RuntimeError(
        "Inference did not complete in time. Please try again in a moment "
        "(the model may still be loading on the server)."
    )

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
