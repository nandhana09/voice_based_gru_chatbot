import { PermissionsAndroid, Platform } from 'react-native';

const requestMicrophonePermission = async () => {
    if (Platform.OS === 'android') {
        const granted = await PermissionsAndroid.request(
            PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
            {
                title: "Microphone Permission",
                message: "This app needs access to your microphone to record audio.",
                buttonNeutral: "Ask Me Later",
                buttonNegative: "Cancel",
                buttonPositive: "OK",
            },
        );
        return (granted === PermissionsAndroid.RESULTS.GRANTED);
    }
    return true; // Assume permission is handled at the app level for iOS
};

export { requestMicrophonePermission };