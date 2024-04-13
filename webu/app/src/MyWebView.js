import React, { useEffect } from 'react';
import { WebView } from 'react-native-webview';
import { View, Text } from 'react-native';
import { requestMicrophonePermission } from './permissions';  // Import from Permissions.js

const MyWebView = () => {
    useEffect(() => {
        const checkPermissions = async () => {
            const hasPermission = await requestMicrophonePermission();
            if (!hasPermission) {
                console.log('Microphone permission denied');
            } else {
                console.log('You can use the microphone');
            }
        };

        checkPermissions();
    }, []);

    return (
        <WebView
            source={{ uri: 'http://192.168.1.6:5000/' }}
            style={{ flex: 1 }}
            mediaPlaybackRequiresUserAction={false}
            javaScriptEnabled={true}
            domStorageEnabled={true}
        />
    );
};

export default MyWebView;