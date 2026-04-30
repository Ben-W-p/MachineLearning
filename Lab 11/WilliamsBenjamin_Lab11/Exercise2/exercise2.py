import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

architectures = [
    {"name": "Model 1", "hidden_layer_sizes": (32,), "activation": "relu", "max_iter": 2000},
    {"name": "Model 2", "hidden_layer_sizes": (64, 32), "activation": "relu", "max_iter": 2000},
    {"name": "Model 3", "hidden_layer_sizes": (128, 64, 32), "activation": "tanh", "max_iter": 3000}
]

results = []

for config in architectures:
    model = MLPRegressor(
        hidden_layer_sizes=config["hidden_layer_sizes"],
        activation=config["activation"],
        solver="adam",
        random_state=42,
        max_iter=config["max_iter"]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(config["name"])
    print("Hidden Layers:", config["hidden_layer_sizes"])
    print("Activation:", config["activation"])
    print("MSE:", mse)
    print("R2:", r2)
    print()

    results.append({
        "Model": config["name"],
        "Hidden Layers": config["hidden_layer_sizes"],
        "Activation": config["activation"],
        "MSE": mse,
        "R2": r2
    })

results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
print(results_df)