import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def load_dataset(file_path):
    data = pd.read_csv(file_path, encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['Label', 'Message']
    return data

def preprocess_data(data):
    data['Label'] = data['Label'].map({'spam': 1, 'ham': 0})

    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['Message'])
    y = data['Label']
    return X, y

def train_svm_model(X_train, y_train):
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Classification Report:\n", classification_report(y_test, predictions))
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    file_path = 'DATA/spam.csv'
    
    data = load_dataset(file_path)
    
    X, y = preprocess_data(data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = train_svm_model(X_train, y_train)
    
    evaluate_model(model, X_test, y_test)
