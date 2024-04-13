import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertModel

# Load data from JSON file
with open('voicebot.json') as file:
    data = json.load(file)

# Extract training sentences, labels, and responses
training_sentences = []
training_labels = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern.lower())  # Convert to lowercase
        training_labels.append(intent['tag'])
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Encode labels
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

# Tokenization and padding
max_len = 20
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sequences = tokenizer(training_sentences, padding='max_length', truncation=True, max_length=max_len, return_tensors='tf')
input_ids = sequences['input_ids']

# Load pre-trained BERT model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Freeze BERT layers
bert_model.trainable = False

# Build model architecture
input_layer = Input(shape=(max_len,), dtype=tf.int32)
bert_output = bert_model(input_layer)[0]
pooling_layer = GlobalAveragePooling1D()(bert_output)
output_layer = Dense(len(labels), activation='softmax')(pooling_layer)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=1e-5), metrics=['accuracy'])

# Model summary
model.summary()

# Train model
epochs = 5
history = model.fit(input_ids, training_labels, epochs=epochs, batch_size=32)

# Save the trained model
model.save('bert_conversational_model')

# Save label encoder and tokenizer
import pickle

with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(lbl_encoder, le_file)

with open('bert_tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

# Example usage for inference
def predict_intent(user_input):
    user_input = user_input.lower()  # Convert to lowercase
    input_sequence = tokenizer(user_input, padding='max_length', truncation=True, max_length=max_len, return_tensors='tf')['input_ids']
    predictions = model.predict(input_sequence)
    predicted_label_id = np.argmax(predictions)
    predicted_label = lbl_encoder.inverse_transform([predicted_label_id])[0]
    return predicted_label

# Example usage
user_input = "Hello, how can I help you today?"
predicted_intent = predict_intent(user_input)
print("Predicted Intent:", predicted_intent)
