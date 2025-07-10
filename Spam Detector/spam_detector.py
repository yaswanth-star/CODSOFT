import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('spam.csv', encoding='latin-1')

df = df[['v1', 'v2']]
df = df.rename(columns={'v1': 'label', 'v2': 'text'})

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train_tfidf, y_train)

y_pred = clf.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

sample = ["Congratulations! You've won a free ticket to Bahamas.",
          "Hey, are we still meeting for lunch?"]
sample_tfidf = vectorizer.transform(sample)
pred_sample = clf.predict(sample_tfidf)

for text, pred in zip(sample, pred_sample):
    print(f"\nMessage: {text}\nPredicted as: {'Spam' if pred == 1 else 'Ham'}")
