import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# ----------------------------
# 0️⃣ Download NLTK resources
# ----------------------------
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ----------------------------
# 1️⃣ Load Dataset
# ----------------------------
df = pd.read_csv("emotion_dataset.csv")

# ----------------------------
# 2️⃣ Preprocess Text
# ----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df['text'] = df['text'].apply(preprocess)

# ----------------------------
# 3️⃣ Features + Labels
# ----------------------------
X = df['text']
y = df['emotion']

# ----------------------------
# 4️⃣ TF-IDF Vectorizer (uni + bi-grams)
# ----------------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_vec = vectorizer.fit_transform(X)

# ----------------------------
# 5️⃣ Train/Test Split (stratified)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 6️⃣ Train Model
# ----------------------------
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# ----------------------------
# 7️⃣ Evaluate Model
# ----------------------------
y_pred = model.predict(X_test)
print("✅ Classification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# 8️⃣ Save Model + Vectorizer
# ----------------------------
with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model & vectorizer saved successfully.\n")

# ----------------------------
# 9️⃣ Interactive Emotion Prediction
# ----------------------------
def predict_emotion(text):
    processed = preprocess(text)
    vec = vectorizer.transform([processed])
    return model.predict(vec)[0]

print("✅ Type a sentence to predict emotion. Type 'exit' to quit.\n")
while True:
    user_input = input("Enter sentence: ")
    if user_input.lower() == "exit":
        print("Exiting...")
        break
    prediction = predict_emotion(user_input)
    print(f"Predicted Emotion: {prediction}\n")

print("✅ Interactive Emotion Prediction completed.")