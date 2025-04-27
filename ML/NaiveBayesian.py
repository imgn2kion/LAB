import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

documents = [
    "debate on healthcare reform.", 
    "Team a on the championship!", 
    "political debate about taxes.", 
    "sports team wins championship.", 
    "healthcare policies and reforms.", 
    "football championship game.",
]

labels = ["politics", "sports", "politics", "sports", "politics", "sports"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

print("Predictions:")
for text, predicted_label in zip(X_test, y_pred):
    original_text = vectorizer.inverse_transform(text)[0]
    print(f"text: '{' '.join(original_text)}' - predicted label: {predicted_label}")
