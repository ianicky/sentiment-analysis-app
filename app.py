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
# IMPROVED STYLING (CLARITY FIXED)
# -------------------------------
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background: #eef2f3;
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #1f4e79;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #333;
    margin-bottom: 25px;
}

/* Labels */
label {
    color: #222 !important;
    font-weight: 500;
}

/* Text area */
textarea {
    background-color: white !important;
    color: black !important;
    border-radius: 10px !important;
    border: 1px solid #ccc !important;
}

/* Button */
div.stButton > button {
    background: linear-gradient(90deg, #1f77b4, #0056b3);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}

/* Result box */
.result {
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    margin-top: 20px;
}

/* Section spacing */
section {
    margin-top: 30px;
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
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="title">💬 Product Review Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze customer reviews using Machine Learning</div>', unsafe_allow_html=True)

# -------------------------------
# SESSION STATE
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# SINGLE REVIEW
# -------------------------------
st.subheader("✍️ Analyze a Review")

review = st.text_area("Enter your review here:")

if st.button("🔍 Analyze Sentiment"):

    if review.strip():
        data = vectorizer.transform([review])
        prediction = model.predict(data)[0]

        # Confidence
        try:
            proba = model.predict_proba(data)[0]
            confidence = round(max(proba) * 100, 2)
        except:
            confidence = "N/A"

        st.session_state.history.append((review, prediction, confidence))

        if prediction == "positive":
            st.markdown(f"<div class='result' style='background:#d4edda;color:#155724;'>✅ Positive 😊<br>Confidence: {confidence}%</div>", unsafe_allow_html=True)

        elif prediction == "negative":
            st.markdown(f"<div class='result' style='background:#f8d7da;color:#721c24;'>❌ Negative 😠<br>Confidence: {confidence}%</div>", unsafe_allow_html=True)

        else:
            st.markdown(f"<div class='result' style='background:#fff3cd;color:#856404;'>⚠️ Neutral 😐<br>Confidence: {confidence}%</div>", unsafe_allow_html=True)

    else:
        st.warning("⚠️ Please enter a review")

# -------------------------------
# BULK ANALYSIS
# -------------------------------
st.markdown("---")
st.subheader("📁 Bulk Analysis (Upload CSV)")

st.info("Upload a CSV file containing a column named 'review'")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "review" in df.columns:
        data = vectorizer.transform(df["review"])
        predictions = model.predict(data)

        df["Sentiment"] = predictions

        st.success("✅ Analysis complete")
        st.dataframe(df)

        st.download_button(
            "⬇️ Download Results",
            df.to_csv(index=False),
            "results.csv"
        )

    else:
        st.error("❌ CSV must contain a 'review' column")

# -------------------------------
# VISUALIZATION
# -------------------------------
st.markdown("---")
st.subheader("📊 Dataset Sentiment Distribution")

try:
    counts = pd.read_csv(counts_path, index_col=0).squeeze()

    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
    ax.set_title("Sentiment Distribution")

    st.pyplot(fig)

except Exception as e:
    st.error(f"Chart error: {e}")

# -------------------------------
# HISTORY
# -------------------------------
st.markdown("---")
st.subheader("🕘 Prediction History")

if st.session_state.history:
    for item in reversed(st.session_state.history):
        st.write(f"📝 {item[0]}")
        st.write(f"➡️ {item[1]} | Confidence: {item[2]}%")
        st.markdown("---")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Developed using Machine Learning (TF-IDF + Logistic Regression)")