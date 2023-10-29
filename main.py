import streamlit as st
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
# Load and preprocess data
def load_and_process_data(folder_path):
    data = []
    labels = []

    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            y, sr = librosa.load(file_path, sr=None)
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            data.append(spectrogram.flatten())
            labels.append(label)

    return np.array(data), np.array(labels)

# Split data
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Train the Random Forest model
def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Streamlit app
def streamlit_app():
    st.title("Audio Classification with Random Forest")

    # File uploader
    audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if audio_file is not None:
        st.audio(audio_file, format='audio/wav', start_time=0)

        if st.button("Classify"):
            # Preprocess the audio file
            y, sr = librosa.load(audio_file, sr=None)
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            processed_audio = spectrogram.flatten()

            # Reshape the input to match the model's expected shape
            processed_audio = np.reshape(processed_audio, (1, -1))

            # Train the Random Forest model (you might want to move this outside the button click event)
            X, y = load_and_process_data(r"C:\College\DPL+EDA+PDS\Audio1")
            X_train, X_test, y_train, y_test = split_data(X, y)
            model = train_random_forest(X_train, y_train)

            # Predict the class using the trained model
            prediction = model.predict(processed_audio)

            st.success(f"Prediction: {prediction[0]}")

if __name__ == "__main__":
    # Run the Streamlit app
    streamlit_app()
