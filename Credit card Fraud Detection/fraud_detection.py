import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('transactions.csv')

print("Sample data:")
print(df.head())


categorical_cols = ['merchant', 'category', 'gender', 'state', 'job', 'city']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

df['dob'] = pd.to_datetime(df['dob'])
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
df['trans_hour'] = df['trans_date_trans_time'].dt.hour

df = df.drop(['cc_num', 'first', 'last', 'street', 'zip', 'trans_num', 'unix_time', 'trans_date_trans_time', 'dob'], axis=1)

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

sample = X_test.iloc[0:1]
print("\nSample prediction:")
print("Predicted:", clf.predict(sample)[0])
print("Actual:", y_test.iloc[0])
