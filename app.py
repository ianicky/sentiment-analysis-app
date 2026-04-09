import streamlit as st
import pickle
import os
import pandas as pd

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

# -------------------------------
# SIMPLE CSS (SAFE - NO OVERRIDES)
# -------------------------------
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    color: #1f4e79;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 20px;
}

.result {
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# PATH SETUP (WORKING VERSION)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "model", "vectorizer.pkl")
counts_path = os.path.join(BASE_DIR, "model", "sentiment_counts.csv")

# -------------------------------
# LOAD MODEL
# -------------------------------
try:
    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------------------
# HEADER (HTML)
# -------------------------------
st.markdown('<div class="title">💬 Product Review Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze customer reviews using Machine Learning</div>', unsafe_allow_html=True)

# -------------------------------
# INPUT
# -------------------------------
review = st.text_area("✍️ Enter a product review:")

if st.button("🔍 Analyze Sentiment"):

    if review.strip() != "":
        data = vectorizer.transform([review])
        prediction = model.predict(data)[0]

        if prediction == "positive":
            st.markdown('<div class="result" style="background:#d4edda;color:#155724;">✅ Positive 😊</div>', unsafe_allow_html=True)

        elif prediction == "negative":
            st.markdown('<div class="result" style="background:#f8d7da;color:#721c24;">❌ Negative 😠</div>', unsafe_allow_html=True)

        else:
            st.markdown('<div class="result" style="background:#fff3cd;color:#856404;">⚠️ Neutral 😐</div>', unsafe_allow_html=True)

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