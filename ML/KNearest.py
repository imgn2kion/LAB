import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = [
    [5.1, 3.5, 1.4, 0.2, "Setosa"],
    [4.9, 3.0, 1.4, 0.2, "Setosa"],
    [4.7, 3.2, 1.3, 0.2, "Setosa"],
    [4.6, 3.1, 1.5, 0.2, "Setosa"],
    [5.0, 3.6, 1.4, 0.2, "Setosa"],
    [5.4, 3.9, 1.7, 0.4, "Setosa"],
    [4.6, 3.4, 1.4, 0.3, "Setosa"],
    [5.0, 3.4, 1.5, 0.2, "Setosa"],
    [5.1, 3.5, 1.4, 0.3, "Setosa"],
    [7.0, 3.2, 4.7, 1.4, "Versicolor"],
    [6.4, 3.2, 4.5, 1.5, "Versicolor"],
    [6.9, 3.1, 4.9, 1.5, "Versicolor"],
    [5.5, 2.3, 4.0, 1.3, "Versicolor"],
    [6.5, 2.8, 4.6, 1.5, "Versicolor"],
    [5.7, 2.8, 4.5, 1.3, "Versicolor"],
    [6.3, 3.3, 6.0, 2.5, "Virginica"],
    [5.8, 2.7, 5.1, 1.9, "Virginica"],
    [7.1, 3.0, 5.9, 2.1, "Virginica"],
    [6.3, 2.9, 5.6, 1.8, "Virginica"]
]

df = pd.DataFrame(data, columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Species"])

X = df.drop(columns=["Species"])
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"K-Nearest Neighbors Classification Accuracy: {accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred, labels=["Setosa", "Versicolor", "Virginica"])
class_report = classification_report(y_test, y_pred, target_names=["Setosa", "Versicolor", "Virginica"])

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)
