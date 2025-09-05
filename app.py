import time, json, requests
import streamlit as st
from huggingface_hub import InferenceClient

# --- Config from secrets ---
MODEL_ID = st.secrets.get("MODEL_ID", "hugps/mh-bert-pt")
HF_TOKEN = st.secrets.get("HF_TOKEN")  # optional for public, recommended

# --- Inference client (no TF/torch needed on Streamlit) ---
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

@st.cache_data(ttl=3600, show_spinner=False)
def ping_model(model_id: str):
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    r = requests.get(url, timeout=30, headers=({"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}))
    return r.status_code, r.json()

def classify(text: str, tries: int = 5, timeout: int = 120):
    """Call HF Inference API, retrying on warm-up/timeouts."""
    payload = {"inputs": text, "parameters": {"return_all_scores": True}, "options": {"wait_for_model": True}}
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    url = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

    for i in range(tries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if r.status_code == 503:  # loading
                time.sleep(3 * (i + 1))
                continue
            if r.status_code == 404:
                raise RuntimeError(f"Model not found at {MODEL_ID}. Check the repo path.")
            r.raise_for_status()
            data = r.json()
            # normalize shape: [[{label,score},...]] -> [{label,score},...]
            if isinstance(data, list) and data and isinstance(data[0], list):
                data = data[0]
            return data
        except requests.exceptions.Timeout:
            time.sleep(3 * (i + 1))
        except requests.exceptions.RequestException as e:
            time.sleep(3 * (i + 1))
    raise RuntimeError("Inference did not complete in time. Please try again.")

def friendly(scores):
    # map labels if your config uses generic LABEL_0/LABEL_1
    out = {}
    for d in scores:
        lab = d.get("label", "")
        pretty = {"LABEL_0": "Non-issue", "LABEL_1": "Potential MH sign"}.get(lab, lab)
        out[pretty] = float(d.get("score", 0.0))
    return out

# --- UI ---
st.set_page_config(page_title="MH Detector", page_icon="🧠")
st.title("🧠 Mental Health Early Signs Detector")
st.caption("Demo — not medical advice.")

status, meta = ping_model(MODEL_ID)
st.caption(f"Ping /api/models/{MODEL_ID} → {status}")
st.code(json.dumps(meta)[:400] + "...", language="json")

text = st.text_area("Enter text", height=140, placeholder="Type or paste text…")
thr  = st.slider("Alert threshold (class 1)", 0.50, 0.90, 0.65, 0.01)

if st.button("Analyze"):
    t = (text or "").strip()
    if len(t) < 3:
        st.warning("Please enter a longer text.")
    else:
        try:
            raw = classify(t)
            scores = friendly(raw)
            p1 = scores.get("Potential MH sign", 0.0)
            label = "Potential MH sign" if p1 >= thr else "Non-issue"
            if label == "Potential MH sign":
                st.error(f"⚠️ {label} — p1={p1:.2f}, thr={thr:.2f}")
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
        raw = classify(s)
        scores = friendly(raw)
        p1 = scores.get("Potential MH sign", 0.0)
        st.write(f"**p1={p1:.2f}** — _{s}_")
