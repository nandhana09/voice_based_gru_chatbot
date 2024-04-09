import json 
import numpy as np 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data from JSON file
with open('voicebot.json') as file:
    data = json.load(file)

# Extract training sentences and labels
training_sentences = []
training_labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])

# Encode labels
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels_encoded = lbl_encoder.transform(training_labels)

# Vectorize text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Using TF-IDF for feature extraction
X_train_tfidf = tfidf_vectorizer.fit_transform(training_sentences).toarray()

# Train Random Forest Classifier
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_tfidf, training_labels_encoded)

# Save the trained model and preprocessing components
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump((model_rf, lbl_encoder, tfidf_vectorizer), file)
