import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

df = pd.read_csv("Student-Pass-Fail.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Decision Tree Confusion Matrix")
plt.show()

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:\n")
print(feature_importance)

new_students = pd.DataFrame({
    'Self_Study_Daily': [2, 9, 5, 1],
    'Tution_Monthly': [45, 28, 20, 30]
})

predictions = model.predict(new_students)
probabilities = model.predict_proba(new_students)

print("\nPredictions for new students:")
print(predictions)

print("\nPrediction Probabilities for new students:")
print(probabilities)