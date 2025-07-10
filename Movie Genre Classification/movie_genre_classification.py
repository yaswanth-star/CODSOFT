import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

with open("train_data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Split by "ID ::: TITLE ::: GENRE ::: DESCRIPTION" repeating pattern
pattern = r'(\d+)\s*:::\s*(.*?)\s*:::\s*(.*?)\s*:::\s*(.*?)(?=\s*\d+\s*:::|$)'
matches = re.findall(pattern, raw_text, re.DOTALL)

df = pd.DataFrame(matches, columns=["ID", "TITLE", "GENRE", "DESCRIPTION"])

print("Sample data:")
print(df.head())

X = df["DESCRIPTION"]
y = df["GENRE"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)


y_pred = clf.predict(X_test_tfidf)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

def predict_genre(description):
    desc_tfidf = vectorizer.transform([description])
    return clf.predict(desc_tfidf)[0]

sample_desc = "A young wizard goes on an adventure to find a magical stone."
print("\nSample prediction:")
print("Description:", sample_desc)
print("Predicted genre:", predict_genre(sample_desc))
