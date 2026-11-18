# Evaluates the classifier on the recorded CSVs.
# This is how we got the accuracy numbers in our report:
# 75/25 stratified train/test split plus 5 fold cross validation.
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

WINDOW_SIZE = 100  # samples per window (~2 seconds at ~51 Hz)


def find_csv(keyword):
    files = [f for f in os.listdir('.') if keyword in f.lower() and f.endswith('.csv')]
    if not files:
        raise SystemExit(f"No CSV with '{keyword}' in the name found in this folder")
    return pd.concat([pd.read_csv(f) for f in files])


def extract_features(df):
    features = []
    data = df[["ax", "ay", "az"]].values

    for start in range(0, len(data) - WINDOW_SIZE, WINDOW_SIZE // 2):  # 50% overlap
        window = data[start:start + WINDOW_SIZE]

        feat = []
        feat.extend(window.mean(axis=0))
        feat.extend(window.std(axis=0))
        feat.extend(window.min(axis=0))
        feat.extend(window.max(axis=0))
        mag = np.sqrt((window**2).sum(axis=1))
        feat.append(mag.mean())
        feat.append(mag.std())
        feat.append(mag.max() - mag.min())

        features.append(feat)

    return np.array(features)


X_rest = extract_features(find_csv("rest"))
X_walk = extract_features(find_csv("walk"))
X_run = extract_features(find_csv("run"))

X = np.vstack([X_rest, X_walk, X_run])
y = np.concatenate([np.zeros(len(X_rest)), np.ones(len(X_walk)), np.full(len(X_run), 2)])

print(f"Total windows: {len(X)}")
print(f"  Rest: {len(X_rest)}, Walking: {len(X_walk)}, Running: {len(X_run)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Rest", "Walking", "Running"]))

scores = cross_val_score(clf, X, y, cv=5)
print(f"Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
