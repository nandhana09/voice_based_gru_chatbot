from flask import Flask, render_template, request, jsonify, session
import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import random
import pickle
from gtts import gTTS
import base64
from io import BytesIO
import speech_recognition as sr

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

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

# Function to convert text to speech using gTTS
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return base64.b64encode(audio_bytes.read()).decode()

# Function to convert speech to text
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak:")
        audio = r.listen(source)
    try:
        user_input = r.recognize_google(audio)
        print("You said:", user_input)
        return user_input
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        return ""

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    user_input = request.form['user_input']
    bot_response = get_bot_response(user_input)
    chat_history.append({"User": user_input, "ChatBot": bot_response})
    print(f"User Input: {user_input}")
    print(f"Bot Response: {bot_response}")
    
    # Convert bot response to speech using gTTS
    audio_data = text_to_speech(bot_response)
    
    return jsonify({'bot_response': bot_response, 'audio_data': audio_data})

@app.route('/process_voice', methods=['POST'])
def process_voice():
    user_input = speech_to_text()
    if user_input:
        session['user_input'] = user_input  # Store recognized text in session
        bot_response = get_bot_response(user_input)
        chat_history.append({"User": user_input, "ChatBot": bot_response})
        print(f"User Input (Voice): {user_input}")
        print(f"Bot Response: {bot_response}")
        
        # Convert bot response to speech using gTTS
        audio_data = text_to_speech(bot_response)
        
        return jsonify({'bot_response': bot_response, 'audio_data': audio_data})
    else:
        return jsonify({'bot_response': 'Could not understand audio', 'audio_data': ''})

@app.route('/chat_history', methods=['GET'])
def chat_history_route():
    return jsonify({'chat_history': chat_history})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
