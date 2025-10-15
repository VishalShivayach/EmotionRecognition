# 🎙️ Text & Speech Emotion Classifier

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-green) ![License](https://img.shields.io/badge/License-MIT-yellow) ![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-orange)

🧠 **AI-powered web app that detects emotions from both text and speech** using state-of-the-art NLP and speech recognition models.  

---

## 🚀 Overview
The **Text & Speech Emotion Classifier** leverages **Hugging Face Transformers** and **Speech Recognition** to detect emotional tone from:  

- ✍️ **Text Input**  
- 🎤 **Live Voice Recording**  
- 📂 **Uploaded Audio Files**  

It identifies emotions like **Joy, Sadness, Anger, Fear, Surprise, Love, and Neutral** — complete with emoji visualization and confidence charts.

---

## 🎬 Demo

![Demo GIF](assets/demo.gif)  
*Example: Live voice input and text emotion detection with interactive bar chart visualization.*

---

## 🧩 Features

- ✅ **Text Emotion Detection** – Instantly analyze the emotion of any sentence  
- 🎙️ **Real-Time Voice Emotion Recognition** – Record live voice input using your microphone  
- 📁 **Audio Upload Support** – Upload `.wav`, `.mp3`, or `.m4a` files for transcription & emotion detection  
- 📊 **Interactive Probability Visualization** – See all detected emotions with confidence bars  
- ⚡ **Powered by Hugging Face Transformers** – Uses `distilroberta-base` fine-tuned for emotion classification  
- 💬 **Speech-to-Text Integration** – Uses SpeechRecognition + Google API for accurate transcription  

---

## 🛠️ Tech Stack

| Category              | Tools / Libraries                                                           |
| --------------------- | --------------------------------------------------------------------------- |
| **Frontend**          | Streamlit                                                                   |
| **NLP Model**         | Hugging Face Transformers (`j-hartmann/emotion-english-distilroberta-base`) |
| **Speech Processing** | SpeechRecognition, pydub, audio-recorder-streamlit                          |
| **Data & Utils**      | Pandas, NumPy                                                               |
| **Language**          | Python 3.8+                                                                 |

---

## 📂 Project Structure
EmotionRecognition/
│
├── app.py # Main Streamlit app
├── requirements.txt # Python dependencies
├── README.md # Documentation
├── assets/ # Screenshots, GIFs, icons, etc.
└── notebooks/ # Jupyter experiments


---

## ⚙️ Installation
1️⃣ **Clone the repository**  
```bash
git clone https://github.com/VishalShivayach/EmotionRecognition.git
cd EmotionRecognition


2️⃣ Create & activate a virtual environment
python -m venv venv
# Mac/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate


3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the app
streamlit run app.py
Access the app at: 👉 http://localhost:8501

🎯 Example Outputs
1. Text Input
   Input: "I just got promoted at work!"
   Prediction: Joy 😀 (Confidence: 98.4%)
2. Live Speech Input
   Record your voice → The app transcribes it and detects emotion automatically.
3. Visualization
  All detected emotions are displayed in an interactive bar chart for comparison.


🧠 Model Details
  1.Model: j-hartmann/emotion-english-distilroberta-base
  2.Architecture: DistilRoBERTa-base fine-tuned for multi-class emotion classification
  3.Supported Emotions: joy, sadness, anger, fear, surprise, neutral, love

🧰 Dependencies
| Library                    | Purpose                    |
| -------------------------- | -------------------------- |
| `streamlit`                | Frontend app framework     |
| `transformers`             | Pre-trained NLP models     |
| `torch`                    | Model backend              |
| `pandas`, `numpy`          | Data handling              |
| `speechrecognition`        | Audio transcription        |
| `pydub`                    | Audio conversion           |
| `audio-recorder-streamlit` | In-browser voice recording |


🧑‍💻 Author
Vishal Shivayach
Made with ❤️ using Streamlit and Hugging Face


⭐ Contributing
  1.Contributions are welcome!
  2.Fork the repository
  3.Create a feature branch: git checkout -b feature/new-feature
  4.Commit your changes
  5.Push to the branch
  6.Open a Pull Request


📜 License
This project is licensed under the MIT License – see the LICENSE file for details.


✅ This version now has:  
- Badge row for Python, Streamlit, Hugging Face, License  
- Demo GIF placeholder for visual example  
- Clean, professional sections with emojis  
- Perfect GitHub README layout  
