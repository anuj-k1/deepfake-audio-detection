import streamlit as st
import librosa
import numpy as np
import pickle
import tempfile
import os

# Load model
with open("deepfake_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🎙️ Deepfake Audio Detector")
st.write("Upload an audio file to check if it is Genuine or Deepfake")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features = np.mean(mfcc.T, axis=0)
        return features
    except:
        return None

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file)

    with st.spinner("Analyzing audio..."):
        features = extract_features(tmp_path)

        if features is not None:
            features = features.reshape(1, -1)
            prediction = model.predict(features)[0]
            confidence = model.predict_proba(features)[0]

            st.markdown("---")
            if prediction == 0:
                st.success(f"✅ GENUINE (Human) Audio")
                st.info(f"Confidence: {confidence[0]*100:.2f}%")
            else:
                st.error(f"🚨 DEEPFAKE (AI-Generated) Audio")
                st.info(f"Confidence: {confidence[1]*100:.2f}%")
        else:
            st.error("Could not process audio file. Try another file.")

    os.unlink(tmp_path)