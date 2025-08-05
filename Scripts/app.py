import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load saved model and vectorizer
with open('Data/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('Data/spam_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("📧 Email Phishing (Spam) Detector")

st.markdown("""
Enter the email content below, and the system will classify it as **SPAM** or **HAM** (legitimate).
""")

user_input = st.text_area("✉️ Enter your email content here:", height=200)

if st.button("🧠 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some email content.")
    else:
        # Transform input text
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction == 1:
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is HAM (legitimate email).")
