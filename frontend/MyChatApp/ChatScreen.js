import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, Platform } from 'react-native';
import axios from 'axios'; // Network library for making requests

const ChatScreen = () => {
  const [userInput, setUserInput] = useState('');
  const [botResponse, setBotResponse] = useState('');
  const [chatHistory, setChatHistory] = useState([]);

  const handleTextChange = (text) => setUserInput(text);

  const handleSendText = async () => {
    if (!userInput) return;

    try {
      const response = await axios.post('http://127.0.0.1:5000/process_text', {
        user_input: userInput,
      });

      setBotResponse(response.data.bot_response);
      setChatHistory([...chatHistory, { User: userInput, ChatBot: response.data.bot_response }]);
      setUserInput(''); // Clear input after sending
    } catch (error) {
      console.error('Error sending text:', error);
      setBotResponse('An error occurred. Please try again later.');
    }
  };

  const handleSpeechInput = async () => {
    // Implement speech recognition logic here (consider using a third-party library)
    // Once you have the recognized text, use it similar to handleSendText
    console.log('Speech recognition not yet implemented.');
  };

  return (
    <View>
      <Text>Chat History:</Text>
      {chatHistory.map((item) => (
        <Text key={item.User}>{item.User}: {item.ChatBot}</Text>
      ))}
      <TextInput
        value={userInput}
        onChangeText={handleTextChange}
        placeholder="Type your message..."
      />
      <Button title="Send Text" onPress={handleSendText} />
      {Platform.OS === 'android' ? ( // Check for Android platform
        <Button title="Use Speech" onPress={handleSpeechInput} disabled={true} />
      ) : null}
      {/* Disable "Use Speech" button on non-Android platforms */}
      <Text>Bot Response: {botResponse}</Text>
    </View>
  );
};

export default ChatScreen;
