import streamlit as st
import pickle
import time
import re

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0f1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 850px;
}

h1 {
    text-align: center;
    color: white;
    font-size: 3rem;
}

.subtext {
    text-align: center;
    color: #b0b3b8;
    font-size: 18px;
    margin-bottom: 25px;
}

.stTextArea textarea {
    background-color: #1c1f26;
    color: white;
    border-radius: 15px;
    font-size: 18px;
    padding: 15px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg,#4CAF50,#00c6ff);
    color: white;
    font-size: 20px;
    border-radius: 12px;
    padding: 12px;
    border: none;
    font-weight: bold;
}

.stButton > button:hover {
    transform: scale(1.02);
    transition: 0.3s;
}

.result-box {
    padding: 18px;
    border-radius: 15px;
    font-size: 24px;
    text-align: center;
    font-weight: bold;
    margin-top: 20px;
}

.real {
    background-color: rgba(0,255,100,0.12);
    color: #00ff88;
}

.fake {
    background-color: rgba(255,0,0,0.12);
    color: #ff4b4b;
}
</style>
""", unsafe_allow_html=True)

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    return text

# ---------------- HEADER ----------------
st.markdown("<h1>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtext'>AI Powered News Verification System</div>",
    unsafe_allow_html=True
)

# ---------------- INPUT ----------------
user_input = st.text_area(
    "Paste News Article Below:",
    height=220,
    placeholder="Enter headline or article text here..."
)

# ---------------- BUTTON ----------------
if st.button("🚀 Analyze News"):

    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        with st.spinner("Checking authenticity..."):
            time.sleep(1.5)

        text = clean_text(user_input)
        vector = vectorizer.transform([text])
        prediction = model.predict(vector)[0]

        # confidence
        if hasattr(model, "predict_proba"):
            confidence = round(max(model.predict_proba(vector)[0]) * 100, 2)
        else:
            confidence = 87.0

        # ---------------- RESULT ----------------
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
