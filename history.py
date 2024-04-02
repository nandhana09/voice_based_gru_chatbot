import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GRU, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data from JSON file
with open('scratch.json') as file:
    data = json.load(file)

# Extract training sentences, labels, and responses
training_sentences = []
training_labels = []
labels = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
        
# Encode labels
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

# Tokenization and padding
vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# Define the number of classes
num_classes = len(labels)

# Build model architecture
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GRU(32, return_sequences=True))  # Add GRU layer with 32 units
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model summary
model.summary()

# Train model
epochs = 500
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

# Save the trained model
model.save("history_gru")

# Save the fitted tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# Save the fitted label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

# Load conversation history (assuming it's stored in a list)
conversation_history = []

# Function to preprocess conversation history
def preprocess_history(conversation_history):
    # Combine past user inputs and bot responses into a single string
    history_text = " ".join([f"{item['user_input']} {item['bot_response']}" for item in conversation_history])
    return history_text

# Function to process user input and conversation history
def process_input(user_input, conversation_history):
    # Preprocess conversation history
    history_text = preprocess_history(conversation_history)
    
    # Combine current user input with conversation history
    input_text = f"{history_text} {user_input}"
    
    # Tokenize and pad input text
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_input = pad_sequences(input_sequence, truncating='post', maxlen=max_len)
    
    return padded_input
