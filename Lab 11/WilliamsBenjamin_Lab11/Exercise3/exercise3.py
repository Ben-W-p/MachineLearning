import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("seeds_dataset.txt", sep=r"\s+", header=None)

df.columns = [
    "area",
    "perimeter",
    "compactness",
    "kernel_length",
    "kernel_width",
    "asymmetry_coefficient",
    "kernel_groove_length",
    "target"
]

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

architectures = [
    {"name": "Model 1", "hidden_layer_sizes": (16,), "activation": "relu", "max_iter": 2000},
    {"name": "Model 2", "hidden_layer_sizes": (32, 16), "activation": "relu", "max_iter": 2500},
    {"name": "Model 3", "hidden_layer_sizes": (64, 32, 16), "activation": "tanh", "max_iter": 3000}
]

results = []

for config in architectures:
    model = MLPClassifier(
        hidden_layer_sizes=config["hidden_layer_sizes"],
        activation=config["activation"],
        solver="adam",
        random_state=42,
        max_iter=config["max_iter"]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(config["name"])
    print("Hidden Layers:", config["hidden_layer_sizes"])
    print("Activation:", config["activation"])
    print("Accuracy:", acc)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

    results.append({
        "Model": config["name"],
        "Hidden Layers": config["hidden_layer_sizes"],
        "Activation": config["activation"],
        "Accuracy": acc
    })

results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print(results_df)