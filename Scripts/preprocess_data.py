import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load dataset
df = pd.read_csv('data/spam_ham_dataset.csv')

# Basic cleanup: lowercase, remove punctuation/numbers
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = text.strip()
    return text

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(df['clean_text'])

# Target variable
y = df['label_num']  # 0 = ham, 1 = spam

# Save processed data & vectorizer
with open('data/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

pd.DataFrame(X.toarray()).to_csv('data/X_vectorized.csv', index=False)
y.to_csv('data/y_labels.csv', index=False)

print("✅ Preprocessing done. Features and labels saved.")
