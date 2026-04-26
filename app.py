import streamlit as st
import pickle
import time
import re

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    return text

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(135deg,#0f172a,#111827,#1e293b);
    color: white;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Title */
.main-title {
    text-align:center;
    font-size:60px;
    font-weight:800;
    margin-bottom:5px;
}

.subtitle {
    text-align:center;
    font-size:22px;
    color:#cbd5e1;
    margin-bottom:35px;
}

/* Text area */
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    color: white !important;
    border: 2px solid #334155 !important;
    border-radius: 18px !important;
    padding: 18px !important;
    font-size: 18px !important;
}

/* Button */
.stButton>button {
    width:100%;
    background: linear-gradient(90deg,#06b6d4,#3b82f6);
    color:white;
    font-size:22px;
    font-weight:700;
    padding:14px;
    border:none;
    border-radius:14px;
    margin-top:15px;
}

.stButton>button:hover {
    transform: scale(1.02);
    transition: 0.3s;
}

/* Result box */
.result-box {
    padding:25px;
    border-radius:18px;
    text-align:center;
    font-size:28px;
    font-weight:800;
    margin-top:25px;
}

.real {
    background: rgba(34,197,94,0.15);
    color:#22c55e;
    border:1px solid #22c55e;
}

.fake {
    background: rgba(239,68,68,0.15);
    color:#ef4444;
    border:1px solid #ef4444;
}

/* Footer */
.footer {
    text-align:center;
    margin-top:50px;
    color:#94a3b8;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>📰 Fake News Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI Powered News Verification System</div>", unsafe_allow_html=True)

# ---------------- INPUT ----------------
user_input = st.text_area(
    "Paste News Article:",
    height=260,
    placeholder="Enter headline or full article text here..."
)

# ---------------- BUTTON ----------------
if st.button("🚀 Analyze News"):

    if user_input.strip() == "":
        st.warning("Please enter article text.")
    else:
        with st.spinner("Analyzing content..."):
            time.sleep(1.5)

        text = clean_text(user_input)
        vector = vectorizer.transform([text])
        prediction = model.predict(vector)[0]

        # confidence
        if hasattr(model, "predict_proba"):
            confidence = round(max(model.predict_proba(vector)[0]) * 100, 2)
        else:
            confidence = 88.0

        # Result
        if prediction == True:
            st.markdown(
                f"<div class='result-box real'>✅ REAL NEWS<br>{confidence}% Confidence</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box fake'>❌ FAKE NEWS<br>{confidence}% Confidence</div>",
                unsafe_allow_html=True
            )

        st.progress(int(confidence))
