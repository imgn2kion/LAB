import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def load_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y, iris.target_names

def train_svm(X_train, y_train):
    model = SVC(kernel="linear", C=1.0, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    svm_model = train_svm(X_train, y_train)
    evaluate_model(svm_model, X_test, y_test, target_names)

    new_samples = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [6.7, 3.1, 4.7, 1.5],
        [7.6, 3.0, 6.6, 2.1]
    ])

    predictions = svm_model.predict(new_samples)

    print("\nPredictions for new samples:")
    for sample, prediction in zip(new_samples, predictions):
        print(f"Sample: {sample}, Predicted Class: {target_names[prediction]}")
