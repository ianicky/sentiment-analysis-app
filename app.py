import streamlit as st
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

# -------------------------------
# STYLING
# -------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #eef2f3, #dfe9f3);
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #1f4e79;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

.result {
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    margin-top: 20px;
}

div.stButton > button {
    background: linear-gradient(90deg, #1f77b4, #0056b3);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# PATH SETUP
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
# HEADER
# -------------------------------
st.markdown('<div class="title">💬 Product Review Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze customer reviews using Machine Learning</div>', unsafe_allow_html=True)

# -------------------------------
# SESSION STATE (HISTORY)
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# SINGLE REVIEW INPUT
# -------------------------------
review = st.text_area("✍️ Enter a product review:")

if st.button("🔍 Analyze Sentiment"):

    if review.strip() != "":
        data = vectorizer.transform([review])
        prediction = model.predict(data)[0]

        # Confidence score
        try:
            proba = model.predict_proba(data)[0]
            confidence = round(max(proba) * 100, 2)
        except:
            confidence = "N/A"

        # Save history
        st.session_state.history.append((review, prediction, confidence))

        # Display result
        if prediction == "positive":
            st.markdown(f'<div class="result" style="background:#d4edda;color:#155724;">✅ Positive 😊<br>Confidence: {confidence}%</div>', unsafe_allow_html=True)

        elif prediction == "negative":
            st.markdown(f'<div class="result" style="background:#f8d7da;color:#721c24;">❌ Negative 😠<br>Confidence: {confidence}%</div>', unsafe_allow_html=True)

        else:
            st.markdown(f'<div class="result" style="background:#fff3cd;color:#856404;">⚠️ Neutral 😐<br>Confidence: {confidence}%</div>', unsafe_allow_html=True)

    else:
        st.warning("⚠️ Please enter a review")

# -------------------------------
# FILE UPLOAD (MULTIPLE REVIEWS)
# -------------------------------
st.markdown("---")
st.subheader("📁 Upload CSV for Bulk Analysis")

uploaded_file = st.file_uploader("Upload CSV file (column: review)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "review" in df.columns:
        data = vectorizer.transform(df["review"])
        predictions = model.predict(data)

        df["Sentiment"] = predictions
        st.write(df)

        st.download_button("Download Results", df.to_csv(index=False), "results.csv")

    else:
        st.error("CSV must contain a 'review' column")

# -------------------------------
# PIE CHART
# -------------------------------
st.markdown("---")
st.subheader("📊 Sentiment Distribution")

try:
    counts = pd.read_csv(counts_path, index_col=0).squeeze()

    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
    ax.axis('equal')
    st.pyplot(fig)

except Exception as e:
    st.error(f"Chart error: {e}")

# -------------------------------
# HISTORY DISPLAY
# -------------------------------
st.markdown("---")
st.subheader("🕘 Prediction History")

if st.session_state.history:
    for item in reversed(st.session_state.history):
        st.write(f"Review: {item[0]}")
        st.write(f"Prediction: {item[1]} | Confidence: {item[2]}%")
        st.markdown("---")

# -------------------------------
# FOOTER
# -------------------------------
st.caption("Developed using Machine Learning (TF-IDF + Logistic Regression)")