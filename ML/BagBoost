import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['Target'] = iris.target

X = df.iloc[:, :-1]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_dt = DecisionTreeClassifier()

bagging_model = BaggingClassifier(estimator=base_dt, n_estimators=10, random_state=42)
bagging_model.fit(X_train, y_train)

y_pred_bagging = bagging_model.predict(X_test)
bagging_accuracy = accuracy_score(y_test, y_pred_bagging)
print("========== Bagging Results ==========")
print("Bagging Accuracy:", bagging_accuracy)
print("Bagging Classification Report:\n", classification_report(y_test, y_pred_bagging))

boosting_model = AdaBoostClassifier(estimator=base_dt, n_estimators=10, random_state=42)
boosting_model.fit(X_train, y_train)

y_pred_boosting = boosting_model.predict(X_test)
boosting_accuracy = accuracy_score(y_test, y_pred_boosting)
print("========== Boosting Results ==========")
print("Boosting Accuracy:", boosting_accuracy)
print("Boosting Classification Report:\n", classification_report(y_test, y_pred_boosting))

results = {
    "Model": ["Bagging (Decision Tree)", "Boosting (Decision Tree)"],
    "Accuracy": [bagging_accuracy, boosting_accuracy]
}

results_df = pd.DataFrame(results)

print("\n========== Final Results Table ==========")
print(results_df)
