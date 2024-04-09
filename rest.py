from flask import Flask, render_template, request, jsonify
import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import random
import pickle
import speech_recognition as sr
from gtts import gTTS
import io

app = Flask(__name__)

# Load data
with open(r'E:\new\dataset\voicebot.json') as file:
    data = json.load(file)

# Load trained model
model = keras.models.load_model(r'E:\new\models\voicebot4.0_gru')

# Load tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load label encoder object
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# Parameters
max_len = 20

# Initialize chat history
chat_history = []

# Function to get bot response
def get_bot_response(user_input):
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]),
                                         truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])
    
    for i in data['intents']:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])

# Function to convert speech to text
def speech_to_text(audio_data):
    recognizer = sr.Recognizer()
    audio = sr.AudioData(bytes(audio_data, 'ISO-8859-1'), sample_rate=16000, sample_width=2)
    try:
        user_input = recognizer.recognize_google(audio)
        return user_input
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        return ""

# Function to convert text to speech using gTTS
def text_to_speech(output):
    audio_bytes = io.BytesIO()
    tts = gTTS(text=output, lang='en')
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    user_input = request.form['user_input']
    bot_response = get_bot_response(user_input)
    chat_history.append({"User": user_input, "ChatBot": bot_response})
    audio_bytes = text_to_speech(bot_response)
    print(f"User Input: {user_input}")
    print(f"Bot Response: {bot_response}")
    return jsonify({'bot_response': bot_response, 'audio': audio_bytes.getvalue().decode('ISO-8859-1')})

@app.route('/speech_to_text', methods=['POST'])
def speech_to_text_route():
    audio_data = request.data
    user_input = speech_to_text(audio_data)
    bot_response = get_bot_response(user_input)
    chat_history.append({"User": user_input, "ChatBot": bot_response})
    audio_bytes = text_to_speech(bot_response)
    print(f"Speech to Text: {user_input}")
    print(f"Bot Response: {bot_response}")
    return jsonify({'user_input': user_input, 'bot_response': bot_response, 'audio': audio_bytes.getvalue().decode('ISO-8859-1')})

# Display chat history
@app.route('/chat_history', methods=['GET'])
def chat_history_route():
    return jsonify({'chat_history': chat_history})

if __name__ == '__main__':
    app.run(debug=True)
