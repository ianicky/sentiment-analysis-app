import streamlit as st
import pickle
import os
import pandas as pd

# -------------------------------
# PAGE CONFIG (MUST BE FIRST)
# -------------------------------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

# -------------------------------
# FORCE FULL UI STYLING (STRONG OVERRIDE)
# -------------------------------
st.markdown("""
<style>

/* FORCE LIGHT MODE */
html, body, [class*="css"]  {
    background-color: #f4f6f8 !important;
    color: #2c3e50 !important;
}

/* Main container */
[data-testid="stAppViewContainer"] {
    background: #f4f6f8 !important;
}

/* Card layout */
[data-testid="stVerticalBlock"] {
    background-color: white !important;
    padding: 30px !important;
    border-radius: 15px !important;
    box-shadow: 0px 4px 25px rgba(0,0,0,0.08) !important;
    margin-bottom: 20px !important;
}

/* Title */
h1 {
    text-align: center;
    font-size: 42px;
}

/* Subtitle */
p {
    text-align: center;
    font-size: 18px;
}

/* Text area */
textarea {
    border-radius: 12px !important;
    border: 2px solid #1f77b4 !important;
    padding: 10px !important;
}

/* Button */
div.stButton > button {
    background: linear-gradient(90deg, #1f77b4, #0056b3);
    color: white !important;
    border-radius: 10px;
    height: 3em;
    font-size: 16px;
    font-weight: bold;
    transition: 0.3s;
}

/* Hover effect */
div.stButton > button:hover {
    transform: scale(1.05);
}

/* Result box */
.result-box {
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# FIXED PATH HANDLING (SAFE VERSION)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "model", "vectorizer.pkl")
counts_path = os.path.join(BASE_DIR, "model", "sentiment_counts.csv")

# -------------------------------
# LOAD FILES SAFELY (PREVENT CRASH)
# -------------------------------
try:
    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
except FileNotFoundError:
    st.error("❌ Model files not found. Check your 'model' folder and file names.")
    st.stop()

# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
<div>
    <h1>💬 Product Review Sentiment Analysis</h1>
    <p>Analyze customer reviews using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# INPUT SECTION
# -------------------------------
review = st.text_area("✍️ Enter a product review:")

if st.button("🔍 Analyze Sentiment"):

    if review.strip() != "":
        data = vectorizer.transform([review])
        prediction = model.predict(data)[0]

        st.subheader("Result:")

        if prediction == "positive":
            st.markdown(
                "<div class='result-box' style='background:#d4edda;color:#155724;'>✅ Positive 😊</div>",
                unsafe_allow_html=True
            )

        elif prediction == "negative":
            st.markdown(
                "<div class='result-box' style='background:#f8d7da;color:#721c24;'>❌ Negative 😠</div>",
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                "<div class='result-box' style='background:#fff3cd;color:#856404;'>⚠️ Neutral 😐</div>",
                unsafe_allow_html=True
            )
    else:
        st.warning("⚠️ Please enter a review")

# -------------------------------
# VISUALIZATION
# -------------------------------
st.markdown("---")
st.subheader("📊 Dataset Sentiment Distribution")

try:
    counts = pd.read_csv(counts_path, index_col=0)
    counts = counts.squeeze()
    st.bar_chart(counts)
except Exception as e:
    st.error(f"Visualization error: {e}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Developed using Machine Learning (TF-IDF + Logistic Regression)")