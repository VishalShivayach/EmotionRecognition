import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from pydub import AudioSegment
import io

# Hugging Face Transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ---------------- Streamlit Page ----------------
st.set_page_config(page_title="Text & Speech Emotion Classifier", page_icon="😊", layout="centered")
st.title("😊 Text & Speech Emotion Classifier")
st.write("Enter text, record your voice, or upload an audio file to detect emotion.")

# ---------------- Emoji Mapping ----------------
emoji_dict = {
    'joy': '😀',
    'sadness': '😢',
    'anger': '😡',
    'fear': '😱',
    'surprise': '😲',
    'neutral': '😐',
    'love': '❤️',
}

# ---------------- Load Hugging Face Model ----------------
@st.cache_resource
def load_emotion_model():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    return classifier

emotion_classifier = load_emotion_model()

# ---------------- Helper: Convert any audio to WAV ----------------
def convert_to_wav(audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    buf.seek(0)
    return buf

# ---------------- Create Tabs ----------------
tab1, tab2, tab3 = st.tabs(["Text Input", "Live Speech Input", "Upload Audio File"])

# ---------------- Text Input ----------------
with tab1:
    text = st.text_area("Your sentence:")
    if st.button("Predict Emotion (Text)"):
        if text.strip() != "":
            results = emotion_classifier(text)
            top_emotion = max(results[0], key=lambda x: x['score'])
            prediction = top_emotion['label']
            confidence = top_emotion['score'] * 100
            emoji = emoji_dict.get(prediction.lower(), "")
            st.success(f"**Predicted Emotion:** {prediction} {emoji}")
            st.info(f"**Confidence:** {confidence:.2f}%")

            # Show probabilities for all emotions
            prob_df = pd.DataFrame({'Emotion': [d['label'] for d in results[0]],
                                    'Probability (%)': [d['score']*100 for d in results[0]]})
            st.subheader("Probability for All Emotions")
            st.bar_chart(prob_df.set_index('Emotion'))
        else:
            st.warning("Please enter some text.")

# ---------------- Live Speech Input ----------------
with tab2:
    st.write("Record your voice:")
    audio_data = audio_recorder()
    if audio_data:
        st.audio(audio_data, format="audio/wav")
        if st.button("Predict Emotion (Live Speech)"):
            r = sr.Recognizer()
            try:
                audio_bytes = convert_to_wav(audio_data)
                with sr.AudioFile(audio_bytes) as source:
                    audio = r.record(source)
                    text_from_speech = r.recognize_google(audio)
                    st.info(f"**Transcribed Text:** {text_from_speech}")

                    results = emotion_classifier(text_from_speech)
                    top_emotion = max(results[0], key=lambda x: x['score'])
                    prediction = top_emotion['label']
                    confidence = top_emotion['score'] * 100
                    emoji = emoji_dict.get(prediction.lower(), "")
                    st.success(f"**Predicted Emotion:** {prediction} {emoji}")
                    st.info(f"**Confidence:** {confidence:.2f}%")

                    prob_df = pd.DataFrame({'Emotion': [d['label'] for d in results[0]],
                                            'Probability (%)': [d['score']*100 for d in results[0]]})
                    st.subheader("Probability for All Emotions")
                    st.bar_chart(prob_df.set_index('Emotion'))

            except Exception as e:
                st.error(f"Error processing audio: {e}")

# ---------------- Upload Audio File ----------------
with tab3:
    st.write("Upload an audio file (.wav, .mp3, .m4a):")
    uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])
    if uploaded_file is not None and st.button("Predict Emotion (Uploaded Audio)"):
        r = sr.Recognizer()
        try:
            audio_bytes = convert_to_wav(uploaded_file.read())
            with sr.AudioFile(audio_bytes) as source:
                audio = r.record(source)
                text_from_speech = r.recognize_google(audio)
                st.info(f"**Transcribed Text:** {text_from_speech}")

                results = emotion_classifier(text_from_speech)
                top_emotion = max(results[0], key=lambda x: x['score'])
                prediction = top_emotion['label']
                confidence = top_emotion['score'] * 100
                emoji = emoji_dict.get(prediction.lower(), "")
                st.success(f"**Predicted Emotion:** {prediction} {emoji}")
                st.info(f"**Confidence:** {confidence:.2f}%")

                prob_df = pd.DataFrame({'Emotion': [d['label'] for d in results[0]],
                                        'Probability (%)': [d['score']*100 for d in results[0]]})
                st.subheader("Probability for All Emotions")
                st.bar_chart(prob_df.set_index('Emotion'))

        except Exception as e:
            st.error(f"Error processing audio: {e}")

# --- Streamlit UI ---
st.write("Made with ❤️ by [Yash Raj](https://github.com/yaahrit)")