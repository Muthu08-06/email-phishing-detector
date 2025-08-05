import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
with open('data/spam_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('data/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Ask user to enter email text
print("\n📨 Enter your email content:")
email_text = input(">>> ")

# Clean text (simple lowercasing)
email_text_cleaned = email_text.lower()

# Vectorize input
X_new = vectorizer.transform([email_text_cleaned])

# Predict
prediction = model.predict(X_new)

# Show result
label = 'SPAM 🚨' if prediction[0] == 1 else 'HAM ✅'
print("\n📢 Prediction:", label)

