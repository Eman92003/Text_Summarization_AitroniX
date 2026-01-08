import os
import requests
import streamlit as st

AR_MODEL_ID = "ahmed0189/mT5-Arabic-text-summarization"
EN_MODEL_ID = "facebook/bart-large-cnn"

st.set_page_config(page_title="Text Summarizer", layout="centered")
st.title("Text Summarizer (Arabic / English)")

# -----------------------------
# Token (Secrets or Env Var)
# -----------------------------
HF_TOKEN = None
try:
    HF_TOKEN = st.secrets.get(HF_TOKEN)
except Exception:
    pass

if not HF_TOKEN:
    HF_TOKEN = os.environ.get(HF_TOKEN)

if not HF_TOKEN:
    st.error("HF_TOKEN is required. Add it to Streamlit Secrets or set it as an environment variable named HF_TOKEN.")
    st.stop()

# -----------------------------
# UI
# -----------------------------
lang = st.selectbox("Select input language", ["Arabic", "English"])
text = st.text_area("Input text", height=260, placeholder="Paste your text here...")

col1, col2 = st.columns(2)
with col1:
    max_new_tokens = st.slider("Max summary length", 30, 250, 120)
with col2:
    min_length = st.slider("Min summary length", 5, 150, 30)

do_summarize = st.button("Summarize", type="primary")

# -----------------------------
# Remote summarization (router endpoint)
# -----------------------------
def hf_summarize(model_id: str, input_text: str) -> str:
    # api-inference.huggingface.co is deprecated -> use router.huggingface.co/hf-inference [web:307][web:310]
    url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": input_text,
        "parameters": {
            # Generation params
            "max_new_tokens": int(max_new_tokens),
            "min_length": int(min_length),
            "do_sample": False,
        },
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)

    # Show useful error details
    if r.status_code != 200:
        try:
            raise RuntimeError(f"HTTP {r.status_code}: {r.json()}")
        except Exception:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")

    data = r.json()

    # Usually: [{"summary_text": "..."}]
    if isinstance(data, list) and data and isinstance(data[0], dict) and "summary_text" in data[0]:
        return data[0]["summary_text"]

    # Sometimes: {"summary_text": "..."}
    if isinstance(data, dict) and "summary_text" in data:
        return data["summary_text"]

    return str(data)

if do_summarize:
    if not text.strip():
        st.warning("Please enter some text first.")
        st.stop()

    model_id = AR_MODEL_ID if lang == "Arabic" else EN_MODEL_ID

    with st.spinner("Summarizing on Hugging Face servers..."):
        try:
            summary = hf_summarize(model_id, text)
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    st.subheader("Summary")
    st.write(summary)
    st.caption(f"Model: {model_id}")

