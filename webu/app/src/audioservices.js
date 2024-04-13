import AudioRecorderPlayer from 'react-native-audio-recorder-player';
import { requestMicrophonePermission } from './permissions';  // Import from Permissions.js

const audioRecorderPlayer = new AudioRecorderPlayer();

const startRecording = async () => {
    const hasPermission = await requestMicrophonePermission();
    if (!hasPermission) {
        console.error('Permission to record not granted');
        return;
    }

    const result = await audioRecorderPlayer.startRecorder();
    audioRecorderPlayer.addRecordBackListener((e) => {
        console.log('Recording:', e.current_position);
    });
    console.log('Start recording:', result);

    // Automatically stop recording after 5 seconds
    setTimeout(async () => {
        const result = await audioRecorderPlayer.stopRecorder();
        audioRecorderPlayer.removeRecordBackListener();
        console.log('Stop recording:', result);
        uploadAudio(result);  // Optional: Upload the recording after stopping
    }, 5000);
};

// uploadAudio function remains the same

export { startRecording, uploadAudio };
