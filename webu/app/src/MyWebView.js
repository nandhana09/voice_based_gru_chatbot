import React from 'react';
import { WebView } from 'react-native-webview';

const MyWebView = () => {
  return <WebView source={{ uri: 'http://192.168.1.3:5000' }} />;
};

export default MyWebView;