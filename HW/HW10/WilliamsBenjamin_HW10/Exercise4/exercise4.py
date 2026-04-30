import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv(
    "lenses.csv",
    header=None,
    names=["age", "spectacle_prescription", "astigmatic", "tear_production_rate", "target"]
)

df["age"] = df["age"].map({1: "young", 2: "pre-presbyopic", 3: "presbyopic"})
df["spectacle_prescription"] = df["spectacle_prescription"].map({1: "myope", 2: "hypermetrope"})
df["astigmatic"] = df["astigmatic"].map({1: "no", 2: "yes"})
df["tear_production_rate"] = df["tear_production_rate"].map({1: "reduced", 2: "normal"})
df["target"] = df["target"].map({1: "hard", 2: "soft", 3: "none"})

X = pd.get_dummies(df.iloc[:, :-1])
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0
)

rf = RandomForestClassifier(n_estimators=500, random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
print("Confusion Matrix:\n")
print(cm)

ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_).plot()
plt.title("Random Forest Confusion Matrix")
plt.show()

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nMost Important Features:\n")
print(feature_importance)