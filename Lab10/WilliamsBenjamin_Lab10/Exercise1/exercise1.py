import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

df = pd.read_csv("Bank-data.csv")

X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    random_state=42
)

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

new_clients = pd.DataFrame({
    'interest_rate': [1.334, 4.857, 0.899],
    'credit': [0, 1, 0],
    'march': [1, 0, 1],
    'may': [0, 1, 1],
    'previous': [0, 1, 3],
    'duration': [250, 600, 150]
})

predictions = model.predict(new_clients)
probabilities = model.predict_proba(new_clients)

print("\nPredictions for new clients:")
print(predictions)

print("\nPrediction Probabilities for new clients:")
print(probabilities)