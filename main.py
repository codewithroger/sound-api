from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib
import uuid
from pydub import AudioSegment
import io

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (important for frontend/API testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and label encoder
model = load_model("Sound_Classifier_CNN.h5")
le = joblib.load("label_encoder.pkl")

# Function to extract features
def extract_features(file_path, target_size=195):
    try:
        audio, sample_rate = librosa.load(file_path, sr=44100)
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

        # Pad or trim to fixed size
        if len(features) < target_size:
            features = np.pad(features, (0, target_size - len(features)), 'constant')
        elif len(features) > target_size:
            features = features[:target_size]

        return features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

# Get metadata (duration/sample rate)
def get_file_metadata(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=audio, sr=sr)
        return {"duration_seconds": duration, "sample_rate": sr}
    except Exception as e:
        return {"error": str(e)}

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_filename = f"{uuid.uuid4().hex}.wav"
    file_path = os.path.join(UPLOAD_FOLDER, temp_filename)

    try:
        contents = await file.read()

        # Handle .mp3 conversion
        if file.filename.lower().endswith(".mp3"):
            audio = AudioSegment.from_file(io.BytesIO(contents), format="mp3")
            audio.export(file_path, format="wav")

        # Handle .wav file
        elif file.filename.lower().endswith((".wav", ".wave")):
            with open(file_path, "wb") as f:
                f.write(contents)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Only .wav and .mp3 are supported.")

        # Extract features
        features = extract_features(file_path)
        if features is None:
            raise HTTPException(status_code=500, detail="Feature extraction failed")

        # Reshape input to match model (assumes Conv2D input)
        input_data = features.reshape(1, -1, 1, 1)

        # Predict
        predictions = model.predict(input_data)
        predicted_index = np.argmax(predictions[0])
        predicted_label = le.inverse_transform([predicted_index])[0]

        metadata = get_file_metadata(file_path)

        return {
            "predicted_label": predicted_label,
            "class_probabilities": predictions[0].tolist(),
            "audio_metadata": metadata
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
