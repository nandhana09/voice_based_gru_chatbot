import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import json

# Load the trained model
model = load_model("lstm2")

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder
with open('label_encoder.pickle', 'rb') as ecn_file:
    lbl_encoder = pickle.load(ecn_file)

# Load dataset from JSON file
with open('voicebot.json') as file:
    dataset = json.load(file)

# Construct responses dictionary
responses_dict = {intent['tag']: intent['responses'] for intent in dataset['intents']}

# Initialize conversation history
conversation_history = []

# Function to preprocess user input
def preprocess_input(user_input, conversation_history):
    max_len = 20
    # Combine past user inputs and bot responses into a single string
    history_text = " ".join([f"{item['user_input']} {item['bot_response']}" for item in conversation_history])
    # Combine current user input with conversation history
    input_text = f"{history_text} {user_input}"
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_input = tf.keras.preprocessing.sequence.pad_sequences(input_sequence, truncating='post', maxlen=max_len)
    return padded_input

# Function to get response based on user input and conversation history
def get_response(user_input, conversation_history):
    processed_input = preprocess_input(user_input, conversation_history)
    predicted_class = np.argmax(model.predict(processed_input), axis=-1)[0]
    predicted_intent = lbl_encoder.inverse_transform([predicted_class])[0]
    responses = responses_dict.get(predicted_intent, ["I'm sorry, I don't understand that."])
    # Add current input and response to conversation history
    conversation_history.append({'user_input': user_input, 'bot_response': responses[0]})
    return responses[0], conversation_history

# Streamlit app
st.title("Voicebot")

user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        response, conversation_history = get_response(user_input, conversation_history)
        st.text_area("Bot:", value=response, height=100)
    else:
        st.warning("Please enter your message.")
