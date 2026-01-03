import os
import re
import tempfile
import numpy as np
import streamlit as st

import torch
import whisper
import nltk
from nltk.tokenize import sent_tokenize

from transformers import BartTokenizer, BartForConditionalGeneration
import evaluate

import spacy


# ----------------------------
# Caching / setup
# ----------------------------
@st.cache_resource
def setup_nlp():
    # Ensure NLTK has a writable data directory on Streamlit Cloud
    nltk_data_dir = os.environ.get("NLTK_DATA", "/tmp/nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)

    # Newer NLTK versions may require both "punkt" and "punkt_tab"
    nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)
    try:
        nltk.download("punkt_tab", download_dir=nltk_data_dir, quiet=True)
    except Exception:
        # Older NLTK versions don't have punkt_tab; ignore.
        pass

    # spaCy model for readability cleanup (optional)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
    return nlp

@st.cache_resource
def load_models(whisper_size: str, hf_model_name: str, device: str):
    w_model = whisper.load_model(whisper_size, device=device)

    tokenizer = BartTokenizer.from_pretrained(hf_model_name)
    model = BartForConditionalGeneration.from_pretrained(hf_model_name).to(device)
    model.eval()
    return w_model, tokenizer, model


# ----------------------------
# Pipeline functions
# ----------------------------
FILLERS = ["uh", "um", "you know", "i mean", "like", "okay", "ok"]
ACTION_KEYWORDS = ["will", "need to", "action", "follow up", "next step"]

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[\d+:\d+:\d+\]", "", text)  # remove timestamps like [00:01:02]
    for f in FILLERS:
        text = re.sub(rf"\b{re.escape(f)}\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_meeting(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[\d+:\d+:\d+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def segment_sentences(text: str):
    """
    Sentence segmentation with NLTK when available.
    Falls back to a simple regex splitter if NLTK data is missing.
    """
    try:
        return sent_tokenize(text)
    except LookupError:
        parts = re.split(r'(?<=[\.\!\?])\s+', text.strip())
        return [p for p in parts if p]

def basic_text_stats(text: str):
    sents = segment_sentences(text) if text.strip() else []
    tokens = text.split() if text.strip() else []
    vocab = set(tokens)

    return {
        "num_sentences": len(sents),
        "num_tokens": len(tokens),
        "vocab_size": len(vocab),
        "avg_sentence_tokens": float(np.mean([len(s.split()) for s in sents])) if sents else 0.0,
        "max_sentence_tokens": int(np.max([len(s.split()) for s in sents])) if sents else 0,
    }

def clean_summary_spacy(summary: str, nlp):
    if nlp is None:
        return summary
    doc = nlp(summary)
    return " ".join([sent.text.strip() for sent in doc.sents])

def extract_action_items(summary: str):
    sents = sent_tokenize(summary) if summary.strip() else []
    return [s for s in sents if any(k in s.lower() for k in ACTION_KEYWORDS)]

def safe_mask_pii(text: str) -> str:
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]", text)
    text = re.sub(r"\b(?:\+?\d[\d\s\-\(\)]{7,}\d)\b", "[PHONE]", text)
    return text

def transcribe_meeting(w_model, media_path: str, device: str):
    result = w_model.transcribe(media_path, fp16=True if device == "cuda" else False)
    return result["text"]

def summarize_meeting(tokenizer, model, device: str, meeting_text: str,
                      max_input: int = 512, max_output: int = 150):
    meeting_text = preprocess_meeting(meeting_text)

    inputs = tokenizer(
        meeting_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=max_output,
            min_length=50,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def rouge_eval(pred: str, ref: str):
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=[pred], references=[ref])


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Meeting Summarizer (Whisper + BART)", layout="wide")
st.title("🎙️ Meeting Summarizer (Whisper + BART)")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.caption(f"Running on: **{device}**")

nlp = setup_nlp()

with st.sidebar:
    st.header("Settings")

    whisper_size = st.selectbox("Whisper model size", ["tiny", "base", "small", "medium", "large"], index=1)
    hf_model_name = st.text_input("Summarization model (HF)", value="facebook/bart-large-cnn")

    st.subheader("Pipeline options")
    use_safe = st.toggle("Safe processing (mask emails/phones)", value=True)
    improve_readability = st.toggle("Improve readability (spaCy sentence cleanup)", value=True)

    st.subheader("Generation limits")
    max_input = st.slider("Max input tokens", 256, 1024, 512, step=64)
    max_output = st.slider("Max output tokens", 60, 256, 150, step=10)

w_model, tokenizer, model = load_models(whisper_size, hf_model_name, device)

tab1, tab2 = st.tabs(["Audio → Summary", "Text → Summary"])

with tab1:
    st.subheader("Upload meeting audio/video")
    media_file = st.file_uploader(
        "Upload a meeting file",
        type=["mp3", "wav", "m4a", "mp4", "mov", "webm"]
    )

    if media_file:
        st.audio(media_file)

        if st.button("1) Transcribe with Whisper"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(media_file.name)[1]) as tmp:
                tmp.write(media_file.read())
                tmp_path = tmp.name

            with st.spinner("Transcribing..."):
                transcript = transcribe_meeting(w_model, tmp_path, device)

            os.unlink(tmp_path)

            if use_safe:
                transcript = safe_mask_pii(transcript)

            st.session_state["transcript"] = transcript

    transcript = st.session_state.get("transcript", "")
    if transcript:
        st.subheader("Transcript")
        st.text_area("Whisper output", transcript, height=220)

        st.subheader("Text statistics")
        cleaned = preprocess_text(transcript)
        st.json(basic_text_stats(cleaned))

        st.subheader("Sentence segmentation (preview)")
        sents = segment_sentences(cleaned)
        st.write(sents[:10])

        if st.button("2) Summarize meeting"):
            with st.spinner("Summarizing..."):
                summ = summarize_meeting(
                    tokenizer, model, device, transcript,
                    max_input=max_input, max_output=max_output
                )

            if improve_readability:
                summ = clean_summary_spacy(summ, nlp)

            if use_safe:
                summ = safe_mask_pii(summ)

            st.session_state["summary"] = summ

    summary = st.session_state.get("summary", "")
    if summary:
        st.subheader("Summary")
        st.success(summary)

        st.subheader("Action items")
        actions = extract_action_items(summary)
        if actions:
            for a in actions:
                st.write("•", a)
        else:
            st.info("No action items detected with current keyword rules.")

        st.subheader("ROUGE evaluation (optional)")
        ref = st.text_area("Paste a reference summary to compute ROUGE (optional)", height=120)
        if ref.strip() and st.button("Compute ROUGE"):
            scores = rouge_eval(summary, ref)
            st.json(scores)

with tab2:
    st.subheader("Paste meeting transcript")
    text_in = st.text_area("Transcript text", height=260)

    if text_in.strip():
        if use_safe:
            text_in = safe_mask_pii(text_in)

        st.subheader("Preprocessed text (preview)")
        cleaned = preprocess_text(text_in)
        st.text_area("Cleaned", cleaned, height=140)

        st.subheader("Text statistics")
        st.json(basic_text_stats(cleaned))

        if st.button("Summarize transcript"):
            with st.spinner("Summarizing..."):
                summ = summarize_meeting(
                    tokenizer, model, device, text_in,
                    max_input=max_input, max_output=max_output
                )

            if improve_readability:
                summ = clean_summary_spacy(summ, nlp)

            if use_safe:
                summ = safe_mask_pii(summ)

            st.subheader("Summary")
            st.success(summ)

            st.subheader("Action items")
            actions = extract_action_items(summ)
            if actions:
                for a in actions:
                    st.write("•", a)
            else:
                st.info("No action items detected.")
