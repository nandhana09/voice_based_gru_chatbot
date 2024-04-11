import React from 'react';
import { NavigationContainer } from '@react-navigation/native'; // Assuming you're using React Navigation
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import ChatScreen from './src/ChatScreen'; // Import the ChatScreen component

const Stack = createNativeStackNavigator();

const App = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Chat" component={ChatScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;
