import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
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

dtree = DecisionTreeClassifier(random_state=0)
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=dtree.classes_)
print("Confusion Matrix:\n")
print(cm)

ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dtree.classes_).plot()
plt.title("Decision Tree Confusion Matrix")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": dtree.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nMost Important Features:\n")
print(feature_importance)

text_representation = export_text(dtree, feature_names=list(X.columns))
print("\nDecision Tree Text Representation:\n")
print(text_representation)

plt.figure(figsize=(16, 10))
plot_tree(dtree, feature_names=X.columns, class_names=dtree.classes_, filled=True)
plt.title("Decision Tree Visualization")
plt.show()