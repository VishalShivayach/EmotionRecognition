import pickle

# Load model + vectorizer
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("🎯 Text Emotion Classifier (type 'quit' to exit)")
while True:
    text = input("Enter a sentence: ")
    if text.lower() == "quit":
        break
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)
    print("Predicted Emotion:", prediction[0])
