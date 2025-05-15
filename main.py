from flask import Flask, request, jsonify
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and label encoder
model = load_model('Sound_Classifier_CNN.h5')
le = joblib.load('Sound_label.pkl')

# Feature extraction
def extract_features(file_name, target_size=195):
    try:
        audio, sample_rate = librosa.load(file_name, sr=44100)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, fmax=8000)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)

        features = np.hstack((
            np.mean(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(mel.T, axis=0),
            np.mean(contrast.T, axis=0),
            np.mean(tonnetz.T, axis=0)
        ))

        if len(features) < target_size:
            features = np.pad(features, (0, target_size - len(features)), 'constant')
        elif len(features) > target_size:
            features = features[:target_size]

        return features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    features = extract_features(file_path)
    if features is None:
        return jsonify({'error': 'Could not extract features'}), 500

    input_data = features.reshape(1, features.shape[0], 1, 1)
    predictions = model.predict(input_data)
    label = le.inverse_transform([np.argmax(predictions)])

    return jsonify({
        'predicted_label': label[0],
        'class_probabilities': predictions.flatten().tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
