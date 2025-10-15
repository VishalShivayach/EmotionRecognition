🎙️ Text & Speech Emotion Classifier
🧠 An AI-powered web app that detects emotions from both text and speech using cutting-edge NLP and speech recognition models.

🚀 Overview
The Text & Speech Emotion Classifier uses Hugging Face Transformers and Speech Recognition to analyze emotional tone from:
  1.✍️ Text Input
  2.🎤 Live Voice Recording
  3.📂 Uploaded Audio Files
It identifies emotions such as Joy, Sadness, Anger, Fear, Surprise, Love, and Neutral — complete with emoji visualization and confidence charts.


🧩 Features
  1.✅ Text Emotion Detection – Type any sentence and get emotion analysis in seconds
  2.🎙️ Real-Time Voice Emotion Recognition – Record live voice input using your microphone
  3.📁 Audio Upload Support – Upload .wav, .mp3, or .m4a files for automatic transcription and emotion detection
  4.📊 Interactive Probability Visualization – Displays all detected emotions with confidence bars
  5.⚡ Powered by Hugging Face Transformers – Uses distilroberta-base fine-tuned for emotion classification
  6.💬 Speech-to-Text Integration – Uses SpeechRecognition + Google API for accurate transcription


  🛠️ Tech Stack
  | Category              | Tools / Libraries                                                           |
| --------------------- | --------------------------------------------------------------------------- |
| **Frontend**          | Streamlit                                                                   |
| **NLP Model**         | Hugging Face Transformers (`j-hartmann/emotion-english-distilroberta-base`) |
| **Speech Processing** | SpeechRecognition, pydub, audio-recorder-streamlit                          |
| **Data & Utils**      | Pandas, NumPy                                                               |
| **Language**          | Python 3.8+                                                                 |


📂 Project Structure
EmotionRecognition/
│
├── app.py                        # Main Streamlit app
├── requirements.txt              # Python dependencies
├── README.md                     # Documentation
├── assets/                       # (optional) screenshots, icons, etc.
└── notebooks/                    # (optional) Jupyter experiments

⚙️ Installation
1️⃣ Clone this Repository
git clone https://github.com/VishalShivayach/EmotionRecognition.git
cd EmotionRecognition


2️⃣ Create & Activate Virtual Environment
python -m venv venv
source venv/bin/activate        # for Mac/Linux
venv\Scripts\activate           # for Windows


3️⃣ Install Dependencies
pip install -r requirements.txt


4️⃣ Run the App
streamlit run app.py
Your app will start at:
👉 http://localhost:8501

🎯 Example Outputs
  1.🧠 Text Input
       Input: “I just got promoted at work!”
       Prediction: Joy 😀 (Confidence: 98.4%)

2.🎤 Live Speech Input
      Record your voice → The app transcribes it and detects emotion automatically.

3. 📊 Visualization
    All detected emotions are displayed in an interactive bar chart for comparison.


🧠 Model Details
  1.Model: j-hartmann/emotion-english-distilroberta-base
  2.Architecture: DistilRoBERTa-base fine-tuned for multi-class emotion classification
  3.Supported Emotions:
    joy, sadness, anger, fear, surprise, neutral, love


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


⭐ Contribute
  1.Contributions are welcome!
  2.Fork the repository
  3.Create a feature branch (git checkout -b feature/new-feature)
  4.Commit your changes
  5.Push and open a PR


📜 License
This project is licensed under the MIT License – see the LICENSE file for details.
