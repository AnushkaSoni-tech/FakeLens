import streamlit as st
import pickle
import time
import re

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fake News Detector", layout="centered")

# ---------------- CLEAN TEXT FUNCTION ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    return text

# ---------------- HEADER ----------------
st.title("📰 Fake News Detection")
st.caption("Real-time AI system to detect misinformation")

# ---------------- INPUT ----------------
user_input = st.text_area("Paste your news article here:", height=200)

# ---------------- ANALYZE ----------------
if st.button("🚀 Analyze News"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(1)

        text = clean_text(user_input)
        vector = vectorizer.transform([text])

        prediction = model.predict(vector)[0]

        # -------- FIXED CONFIDENCE --------
        if hasattr(model, "predict_proba"):
            confidence = round(max(model.predict_proba(vector)[0]) * 100, 2)

        elif hasattr(model, "decision_function"):
            score = model.decision_function(vector)[0]

            # convert score into confidence %
            confidence = round((1 / (1 + abs(score))) * 100, 2)

        else:
            confidence = 85.0

        # -------- RESULT --------
        if prediction == True:
            st.success(f"✅ REAL News ({confidence}%)")
        else:
            st.error(f"❌ FAKE News ({confidence}%)")

        st.progress(int(confidence))
        st.caption(f"Confidence: {confidence}%")