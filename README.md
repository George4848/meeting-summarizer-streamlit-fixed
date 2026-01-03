# Meeting Summarizer (Streamlit + Whisper + BART)

A simple Streamlit web app that:
- Uploads meeting audio/video and transcribes using **Whisper**
- Cleans and preprocesses text
- Shows sentence segmentation + basic text statistics
- Summarizes meetings using a HuggingFace summarization model (default: **facebook/bart-large-cnn**)
- Extracts simple action items (keyword-based)
- Optional ROUGE evaluation (if you paste a reference summary)
- Optional "safe processing" masking for emails/phones

## 1) Run locally

### Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Run
```bash
streamlit run app.py
```

> Whisper needs **ffmpeg** installed on your system.  
> On Ubuntu: `sudo apt-get update && sudo apt-get install -y ffmpeg`

## 2) Deploy on Streamlit Community Cloud (GitHub)

1. Push this repo to GitHub.
2. Go to Streamlit Community Cloud → **New app**
3. Select your GitHub repo and branch
4. Set:
   - **Main file path**: `app.py`

### Notes for deployment
- `packages.txt` installs system dependency **ffmpeg** on Streamlit Cloud.
- App uses caching so models load once per session.

## Repo structure
```
.
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
└── .gitignore
```
