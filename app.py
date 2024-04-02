import streamlit as st
import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import random
import pickle
import speech_recognition as sr
from gtts import gTTS
import io

# Load data
with open("scratch.json") as file:
    data = json.load(file)

# Load trained model
model = keras.models.load_model('history_gru')

# Load tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load label encoder object
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# Parameters
max_len = 20

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to get bot response
def get_bot_response(user_input):
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]),
                                         truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])
    
    for i in data['intents']:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])

# Function to convert speech to text
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Speak:")
        audio = r.listen(source)
    try:
        user_input = r.recognize_google(audio)
        st.write("You:", user_input)
        return user_input
    except sr.UnknownValueError:
        st.write("Could not understand audio")
        return ""
    except sr.RequestError as e:
        st.write("Could not request results; {0}".format(e))
        return ""

# Function to convert text to speech using gTTS
def text_to_speech(output):
    audio_bytes = io.BytesIO()
    tts = gTTS(text=output, lang='en')
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

# Streamlit UI
st.title("ChatBot")

# user_input = st.text_input("You: ", "")

if st.button("Speak"):
    user_input = speech_to_text().strip()
    st.session_state.user_input = user_input  # Store converted text
    st.write("Recognized Text:", user_input)  # Debugging
else:
    user_input = st.text_input("You: ", "")
    # Check if converted text exists in session state
    if "user_input" in st.session_state:
        user_input = st.session_state.user_input

if st.button("Send"):
    if user_input.lower() == "quit":
        st.text("Chat ended.")
    else:
        bot_response = get_bot_response(user_input)
        # Append current chat to chat history
        st.session_state.chat_history.append({"User": user_input, "ChatBot": bot_response})
        st.text_area("ChatBot:", value=bot_response, height=100)
        audio_bytes = text_to_speech(bot_response)
        st.audio(audio_bytes, format='audio/mp3', start_time=0)  # Added start_time=0 to start audio automatically

# Display chat history
st.title("Chat History")
for chat in st.session_state.chat_history:
    st.text(f"You: {chat['User']}")
    st.text(f"ChatBot: {chat['ChatBot']}")
    st.text("")  # Add an empty line between chats
