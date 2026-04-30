import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("balloons_extended.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X = pd.get_dummies(X)

y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=y_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n")
print(cm)

ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_encoder.classes_).plot()
plt.title("Random Forest Confusion Matrix")
plt.show()

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nMost Important Features:\n")
print(feature_importance)