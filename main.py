# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the IMDb movie reviews dataset
# Replace 'path_to_dataset' with the actual path to your dataset file
data = pd.read_csv('C:\Users\sohai\OneDrive\Documents\All_Projects\ML_Project\IMDB Dataset.csv', header=None, names=['review', 'sentiment'])

# Preprocessing
# Convert sentiment labels to numerical values (0 for negative, 1 for positive)
data['sentiment'] = data['sentiment'].map({'negative': 0, 'positive': 1})

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # You can adjust max_features based on your dataset size
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Model training
svm_classifier = LinearSVC()
svm_classifier.fit(X_train_tfidf, y_train)

# Predictions
y_pred = svm_classifier.predict(X_test_tfidf)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
