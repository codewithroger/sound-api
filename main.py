from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib
import shutil
import uuid
from pydub import AudioSegment
import io

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and label encoder once at startup
model = load_model("Sound_Classifier_CNN.h5")
le = joblib.load("label_encoder.pkl")

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

        # Pad or truncate features to fixed length
        if len(features) < target_size:
            features = np.pad(features, (0, target_size - len(features)), 'constant')
        elif len(features) > target_size:
            features = features[:target_size]

        return features

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def get_file_metadata(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=audio, sr=sr)
        return {"duration_seconds": duration, "sample_rate": sr}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Generate unique temp filename
    temp_filename = f"{uuid.uuid4().hex}.wav"
    file_path = os.path.join(UPLOAD_FOLDER, temp_filename)

    try:
        # Read and save file depending on extension
        filename_lower = file.filename.lower()
        if filename_lower.endswith(".mp3"):
            # Convert mp3 to wav using pydub
            audio_data = await file.read()
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
            audio.export(file_path, format="wav")

        elif filename_lower.endswith(".wav") or filename_lower.endswith(".wave"):
            # Save wav directly
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Only .wav and .mp3 are supported.")

        # Extract features
        features = extract_features(file_path)
        if features is None:
            raise HTTPException(status_code=500, detail="Feature extraction failed")

        # Prepare input shape for the model
        input_data = features.reshape(1, features.shape[0], 1, 1)

        # Predict
        predictions = model.predict(input_data)
        predicted_label = le.inverse_transform(np.argmax(predictions, axis=1))[0]
        class_probabilities = predictions.flatten().tolist()
        metadata = get_file_metadata(file_path)

        return {
            "predicted_label": predicted_label,
            "class_probabilities": class_probabilities,
            "audio_metadata": metadata
        }

    except Exception as e:
        # Return error details as HTTPException for client
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp file if exists
        if os.path.exists(file_path):
            os.remove(file_path)
