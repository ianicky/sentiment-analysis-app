import streamlit as st
import pickle
import os
import pandas as pd

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
# TITLE
# -------------------------------
st.title("💬 Product Review Sentiment Analysis System")
st.markdown("Analyze customer reviews using Machine Learning")

# -------------------------------
# USER INPUT
# -------------------------------
review = st.text_area("✍️ Enter a product review:")

if st.button("🔍 Analyze Sentiment"):

    if review.strip() != "":
        data = vectorizer.transform([review])
        prediction = model.predict(data)[0]

        st.subheader("Result:")

        if prediction == "positive":
            st.success("✅ Positive 😊")
        elif prediction == "negative":
            st.error("❌ Negative 😠")
        else:
            st.warning("⚠️ Neutral 😐")
    else:
        st.warning("⚠️ Please enter a review")

# -------------------------------
# VISUALIZATION (FINAL 🔥)
# -------------------------------
st.markdown("---")
st.subheader("📊 Dataset Sentiment Distribution")

try:
    # Load precomputed sentiment counts
    counts = pd.read_csv(counts_path, index_col=0)

    # Convert to proper format
    counts = counts.squeeze()

    # Display chart
    st.bar_chart(counts)

except Exception as e:
    st.error(f"Visualization error: {e}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Developed using Machine Learning (TF-IDF + Logistic Regression)")