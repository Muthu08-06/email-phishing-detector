import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/spam_ham_dataset.csv')

# Print sample records
print("📄 Sample Emails:")
print(df[['label', 'text']].head(5))

# Print label distribution
print("\n📊 Label Distribution:")
print(df['label'].value_counts())

# Plot the class distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label', palette='Set2')
plt.title("Ham vs Spam Email Distribution")
plt.xlabel("Email Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
