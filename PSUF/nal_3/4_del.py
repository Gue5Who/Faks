import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasRegressor
import matplotlib.pyplot as plt
import os


# ============================================================
# 1) LOAD DATA
# ============================================================
I = np.load("DataK/intensity505noise0.npy")
K = np.load("DataK/Kvalues.npy").astype("float32")

print("Loaded shapes:", I.shape, K.shape)


# ============================================================
# 2) SCALE INPUT AND OUTPUT
# ============================================================
# scale X
scaler_X = StandardScaler()
X_s = scaler_X.fit_transform(I)

# scale K because values are extremely small (~1e-11)
K_scale = 1e-11
y_s = K / K_scale


# ============================================================
# 3) MODEL BUILDER FOR GRID SEARCH
# ============================================================
def build_model(activation="relu", lr=1e-3, momentum=0.0, optimizer_name="adam"):
    
    model = models.Sequential([
        layers.Input(shape=(X_s.shape[1],)),
        layers.Dense(128, activation=activation),
        layers.Dense(64, activation=activation),
        layers.Dense(1)
    ])

    if optimizer_name == "adam":
        opt = optimizers.Adam(learning_rate=lr)
    elif optimizer_name == "sgd":
        # Learning rate schedule (decay)
        lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=5000,
            decay_rate=0.96
        )
        opt = optimizers.SGD(
            learning_rate=lr_schedule,
            momentum=momentum
        )
    else:
        raise ValueError("Unknown optimizer")

    model.compile(
        optimizer=opt,
        loss="mse",
        metrics=["mae"]
    )
    return model


# ============================================================
# 4) WRAP MODEL FOR GRID SEARCH
# ============================================================
reg = KerasRegressor(
    model=build_model,
    verbose=0
)

# ============================================================
# 5) DEFINE HYPERPARAMETER GRID
# ============================================================
param_grid = {
    "model__activation": ["relu"],
    "model__optimizer_name": ["adam", "sgd"],
    "model__lr": [1e-3, 1e-4],
    "model__momentum": [0.0, 0.9],   # only for SGD
    "batch_size": [64, 128],
    "epochs": [60]
}

# ============================================================
# 6) K-FOLD CROSS VALIDATION SETUP
# ============================================================
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=reg,
    param_grid=param_grid,
    cv=kfold,
    scoring="neg_mean_squared_error",
    verbose=2
)


# ============================================================
# 7) RUN GRID SEARCH
# ============================================================
print("Running hyperparameter search... (this may take a while)")

grid_result = grid.fit(X_s, y_s)

print("\nBest score:", grid_result.best_score_)
print("Best params:", grid_result.best_params_)


# ============================================================
# 8) RETRAIN BEST MODEL ON FULL DATA
# ============================================================
print("\nRetraining best model on full dataset...")

best_params = grid_result.best_params_
best_model = build_model(
    activation = best_params["model__activation"],
    lr         = best_params["model__lr"],
    momentum   = best_params["model__momentum"],
    optimizer_name = best_params["model__optimizer_name"]
)

history = best_model.fit(
    X_s, y_s,
    epochs = best_params["epochs"],
    batch_size = best_params["batch_size"],
    verbose=1
)


#Best score: -0.025312219560146332
#Best params: {'batch_size': 64, 'epochs': 60, 'model__activation': 'relu', 'model__lr': 0.001, 'model__momentum': 0.9, 'model__optimizer_name': 'adam'}


# ============================================================
# 9) PLOT TRAINING HISTORY
# ============================================================
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Trening MSE")
plt.plot(history.history["mae"], label="Trening MAE")
plt.xlabel("Epoka")
plt.ylabel("Napaka")
plt.title("Učenje najboljšega modela po GridSearch")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.getcwd() + "/figs/best_model_training_curve.pdf", format = 'pdf')
plt.show()

print("DONE.")
