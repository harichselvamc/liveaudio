import streamlit as st
import tensorflow as tf
import tensorflow_io as tfio
import pyaudio
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Load the audio classification model
classification_model = tf.keras.models.load_model('scream_classifier_mlp_TEST_cnn.h5')

# Initialize audio input stream
p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=16000)

st.title("Live Audio Classifier")

# Create a text box for displaying the detected label
detected_label = st.empty()

# Create an empty list to store prediction values
prediction_values = []

# Create a placeholder for the table
prediction_table = st.empty()

# Create a placeholder for the real-time spectrogram
realtime_spectrogram = st.empty()

# Create a placeholder for the positive scream spectrogram
positive_scream_spectrogram = st.empty()

# Create a placeholder for the audio length
audio_length_display = st.empty()

# Create a placeholder for the audio histogram
audio_histogram = st.empty()

# Create a placeholder for the audio waveform
audio_waveform = st.empty()

# Create a placeholder for the MFCCs plot
mfccs_plot = st.empty()

# Create a placeholder for the pitch contour plot
pitch_contour_plot = st.empty()

# Create a placeholder for the chromagram plot
chromagram_plot = st.empty()

# Create a placeholder for the spectrogram plot
spectrogram_plot = st.empty()

# Create a placeholder for the time-domain signal plot
time_domain_plot = st.empty()

# Create a placeholder for the spectral contrast plot
spectral_contrast_plot = st.empty()

# Create a placeholder for the chroma feature plot
chroma_plot = st.empty()

# Create a placeholder for the polyphonic CQT plot
poly_cqt_plot = st.empty()

# Create a placeholder for the beat detection plot
beat_plot = st.empty()

while True:
    # Read audio data
    audio_data = stream.read(16000)
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    audio_array = tf.cast(audio_array, tf.float32) / 32767.0  # Normalize audio

    # Get the audio length in seconds
    audio_length = len(audio_array) / 16000

    # Reshape audio
    audio_array = tf.reshape(audio_array, (1, 16000))

    # Predict
    prediction = classification_model.predict(audio_array)

    # Get the current time
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    prediction_value = float(prediction[0][0])

    if prediction_value > 0.5:
        detected_label.text("Positive Scream Detected")

        # Conditionally add the data to the table only when "Positive Scream Detected"
        prediction_values.append({"Time": current_time, "Prediction Value": prediction_value})
        prediction_table.table(pd.DataFrame(prediction_values, columns=["Time", "Prediction Value"]))

        # Create a Mel spectrogram for positive scream
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_array[0].numpy(), sr=16000)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', cmap='coolwarm')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Positive Scream Mel Spectrogram")
        positive_scream_spectrogram.pyplot(plt)
    else:
        detected_label.text("No Scream Detected")

    # Display the real-time Mel spectrogram
    mel_spectrogram_realtime = librosa.feature.melspectrogram(y=audio_array[0].numpy(), sr=16000)
    mel_spectrogram_db_realtime = librosa.power_to_db(mel_spectrogram_realtime, ref=np.max)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mel_spectrogram_db_realtime, x_axis='time', y_axis='mel', cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Real-time Mel Spectrogram")
    realtime_spectrogram.pyplot(plt)

    # Create an audio histogram
    plt.figure(figsize=(10, 6))
    plt.hist(audio_array[0].numpy(), bins=50, color='blue', alpha=0.7)
    plt.title("Audio Histogram")
    audio_histogram.pyplot(plt)

    # Create an audio waveform plot
    plt.figure(figsize=(10, 3))
    plt.plot(audio_array[0].numpy(), color='blue')
    plt.title("Audio Waveform")
    audio_waveform.pyplot(plt)

    # Create MFCCs plot
    mfccs = librosa.feature.mfcc(y=audio_array[0].numpy(), sr=16000, n_mfcc=13)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mfccs, x_axis='time', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title("MFCCs")
    mfccs_plot.pyplot(plt)

    # Create pitch contour plot
    pitches, magnitudes = librosa.piptrack(y=audio_array[0].numpy(), sr=16000)
    pitches = pitches[pitches > 0]  # Filter out zero pitches
    plt.figure(figsize=(10, 3))
    plt.plot(pitches, color='blue')
    plt.title("Pitch Contour")
    pitch_contour_plot.pyplot(plt)

    # Create chromagram plot
    chromagram = librosa.feature.chroma_stft(y=audio_array[0].numpy(), sr=16000)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')
    plt.title("Chromagram")
    chromagram_plot.pyplot(plt)

    # Create a spectrogram plot
    plt.figure(figsize=(10, 6))
    spectrogram = np.abs(librosa.stft(audio_array[0].numpy()))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), y_axis='log', x_axis='time', cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")
    spectrogram_plot.pyplot(plt)

    # Create a time-domain signal plot
    plt.figure(figsize=(10, 3))
    plt.plot(audio_array[0].numpy(), color='blue')
    plt.title("Time-Domain Signal")
    time_domain_plot.pyplot(plt)

    # Create a spectral contrast plot
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_array[0].numpy(), sr=16000)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(spectral_contrast, x_axis='time', cmap='viridis')
    plt.title("Spectral Contrast")
    spectral_contrast_plot.pyplot(plt)

    # Create a chroma feature plot
    chroma = librosa.feature.chroma_cqt(y=audio_array[0].numpy(), sr=16000)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', cmap='coolwarm')
    plt.title("Chroma Feature")
    chroma_plot.pyplot(plt)

    # Create a polyphonic CQT plot
    poly_cqt = librosa.hybrid_cqt(audio_array[0].numpy(), sr=16000)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(poly_cqt, ref=np.max), x_axis='time', y_axis='cqt_hz', cmap='coolwarm')
    plt.title("Polyphonic CQT")
    poly_cqt_plot.pyplot(plt)

    # Create a beat detection plot
    onset_env = librosa.onset.onset_strength(y=audio_array[0].numpy(), sr=16000)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=16000)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env)
    beats = librosa.frames_to_time(onset_frames, sr=16000)
    plt.figure(figsize=(10, 3))
    plt.plot(beats, onset_env[onset_frames], 'ro')
    plt.title(f"Beat Detection (Tempo: {tempo:.2f} BPM)")
    beat_plot.pyplot(plt)

    # Display the audio length
    audio_length_display.text(f"Audio Length: {audio_length:.2f} seconds")

    # Limit the number of stored prediction values (for better performance)
    max_prediction_values = 100
    if len(prediction_values) > max_prediction_values:
        prediction_values = prediction_values[-max_prediction_values:]
