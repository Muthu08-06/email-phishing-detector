import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load vectorized features and labels
X = pd.read_csv('data/X_vectorized.csv')
y = pd.read_csv('data/y_labels.csv').squeeze()  # make it a 1D series

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("✅ Model Training Complete")
print("🔍 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
with open('data/spam_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

print("💾 Model saved as spam_classifier.pkl")
