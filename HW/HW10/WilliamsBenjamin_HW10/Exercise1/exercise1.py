import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
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

dtree = DecisionTreeClassifier(random_state=0)
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=y_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n")
print(cm)

ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_encoder.classes_).plot()
plt.title("Decision Tree Confusion Matrix")
#plt.show()

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
plot_tree(
    dtree,
    feature_names=X.columns,
    class_names=y_encoder.classes_,
    filled=True
)
plt.title("Decision Tree Visualization")
plt.show()