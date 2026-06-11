# spam_detector.py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training Data
emails = [
    "Congratulations you won a free iPhone",
    "Claim your free prize now",
    "Win cash rewards today",
    "Meeting is scheduled for tomorrow",
    "Please send the report",
    "Project deadline is next week",
    "Let's have lunch together",
    "Can you attend the meeting?"
]

labels = [
    "spam",
    "spam",
    "spam",
    "ham",
    "ham",
    "ham",
    "ham",
    "ham"
]

# Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Train model
model = MultinomialNB()
model.fit(X, labels)

print("=== Spam Email Detector ===")

while True:
    message = input("\nEnter an email message (or type 'exit'): ")

    if message.lower() == "exit":
        print("Program terminated.")
        break

    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)

    print("Prediction:", prediction[0].upper())