import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("/nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# df = pd.read_csv("breastcancer.csv")

# df = df.drop(columns=["Unnamed: 32"], errors="ignore")

# df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# X = df.drop(["id", "diagnosis"], axis=1)

# y = df["diagnosis"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42
# )

# model = GaussianNB()

# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# cm = confusion_matrix(y_test, y_pred)

# print("Confusion Matrix:")
# print(cm)

# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")

# plt.show()