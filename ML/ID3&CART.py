import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv('DATA/ID3.csv')

print("First 5 rows of data:\n", data.head())

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("\nFeatures (X) - first 5 rows:\n", X.head())
print("\nTarget (y) - first 5 rows:\n", y.head())

le_outlook = LabelEncoder()
le_temp = LabelEncoder()
le_humid = LabelEncoder()
le_wind = LabelEncoder()

X.loc[:,'Outlook'] = le_outlook.fit_transform(X['Outlook'])
X.loc[:,'Temp'] = le_temp.fit_transform(X['Temp'])
X.loc[:,'Humidity'] = le_humid.fit_transform(X['Humidity'])
X.loc[:,'Wind'] = le_wind.fit_transform(X['Wind'])

print("\nTransformed Features (X):\n", X.head())

le_decision = LabelEncoder()
y = le_decision.fit_transform(y)

print("\nTransformed Target (y):\n", y)

clf = DecisionTreeClassifier(criterion='')
clf.fit(X, y)

def encode_input(inp):
    inp[0] = le_outlook.transform([inp[0]])[0]
    inp[1] = le_temp.transform([inp[1]])[0]
    inp[2] = le_humid.transform([inp[2]])[0]
    inp[3] = le_wind.transform([inp[3]])[0]
    return [inp]

new_inp = ["Sunny", "Cool", "High", "Strong"]

enc_inp = encode_input(new_inp)

enc_inp_df = pd.DataFrame(enc_inp, columns=X.columns)

y_pred = clf.predict(enc_inp_df)

print("\nPrediction for input {}: {}".format(new_inp, le_decision.inverse_transform(y_pred)[0]))

plt.figure(figsize=(10, 8))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=le_decision.classes_,
    filled=True
)
plt.title("Decision Tree Visualization")
plt.show()
