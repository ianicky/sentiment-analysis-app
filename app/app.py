import streamlit as st
import pickle
import os
import pandas as pd

# -------------------------------
# CUSTOM CSS (HTML + CSS INJECTION)
# -------------------------------
st.markdown("""
<style>

/* Page background */
body {
    background-color: #f4f6f8;
}

/* Main container */
.main {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
}

/* Title styling */
h1 {
    color: #2c3e50;
    text-align: center;
}

/* Text area */
textarea {
    border-radius: 10px !important;
    border: 1px solid #ccc !important;
    padding: 10px !important;
}

/* Button */
div.stButton > button {
    background-color: #007bff;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

/* Button hover */
div.stButton > button:hover {
    background-color: #0056b3;
    color: white;
}

/* Result box */
.result-box {
    padding: 15px;
    border-radius: 10px;
    margin-top: 20px;
    text-align: center;
    font-weight: bold;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# FIXED PATH HANDLING
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, '..', 'model', 'model.pkl')
vectorizer_path = os.path.join(BASE_DIR, '..', 'model', 'vectorizer.pkl')
counts_path = os.path.join(BASE_DIR, '..', 'model', 'sentiment_counts.csv')

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

# -------------------------------
# HEADER (HTML)
# -------------------------------
st.markdown("""
<div style='text-align: center;'>
    <h1>💬 Product Review Sentiment Analysis</h1>
    <p>Analyze customer reviews using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# USER INPUT
# -------------------------------
review = st.text_area("✍️ Enter a product review:")

if st.button("🔍 Analyze Sentiment"):

    if review.strip() != "":
        data = vectorizer.transform([review])
        prediction = model.predict(data)[0]

        st.subheader("Result:")

        # Styled result output
        if prediction == "positive":
            st.markdown(
                "<div class='result-box' style='background-color:#d4edda;color:#155724;'>✅ Positive 😊</div>",
                unsafe_allow_html=True
            )

        elif prediction == "negative":
            st.markdown(
                "<div class='result-box' style='background-color:#f8d7da;color:#721c24;'>❌ Negative 😠</div>",
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                "<div class='result-box' style='background-color:#fff3cd;color:#856404;'>⚠️ Neutral 😐</div>",
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

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.bar_chart(counts)
    st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Visualization error: {e}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Developed using Machine Learning (TF-IDF + Logistic Regression)")