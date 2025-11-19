import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =====================================================
# 1) Load data
# =====================================================
cwd = os.getcwd()
data_dir = os.path.join(cwd, "DataK")

print("Loading data from:", data_dir)

I = np.load(os.path.join(data_dir, "intensity505noise0.npy")).astype("float32")
K = np.load(os.path.join(data_dir, "Kvalues.npy")).astype("float32")

# Train / val / test split
X_train, X_temp, y_train, y_temp = train_test_split(
    I, K, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("Shapes:")
print("  Train:", X_train.shape)
print("  Val:  ", X_val.shape)
print("  Test: ", X_test.shape)

# Normalize X
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)


# =====================================================
# 2) Hyperparameter: different learning rates
# =====================================================
learning_rates = [1e-2, 1e-3, 1e-4]
histories_lr = {}

print("\nTraining models with different learning rates...\n")


# =====================================================
# 3) Model builder helper
# =====================================================
def build_model_lr(lr):
    model = models.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )
    return model


# =====================================================
# 4) Train models for each learning rate
# =====================================================
for lr in learning_rates:
    print(f"Training with learning rate = {lr}")
    model_lr = build_model_lr(lr)
    history_lr = model_lr.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=120,
        batch_size=128,
        verbose=0
    )
    histories_lr[lr] = history_lr.history


# =====================================================
# 5) Plot results
# =====================================================
plt.figure(figsize=(9,6))

colors = ['tab:blue', 'tab:orange', 'tab:green']

for (lr, c) in zip(learning_rates, colors):
    plt.plot(histories_lr[lr]['loss'],      color=c, linestyle='-',  label=f"Trening (lr={lr})")
    plt.plot(histories_lr[lr]['val_loss'], color=c, linestyle='--', label=f"Validacija (lr={lr})")

plt.xlabel("Epoka")
plt.ylabel("Napaka (MSE)")
plt.title("Vpliv hitrosti učenja (learning rate) na potek učenja", fontsize=14)
plt.grid(True)
plt.ylim(0, 0.002)
plt.legend()
plt.tight_layout()

# Save PDF to figs/
os.makedirs("figs", exist_ok=True)
plt.savefig(os.getcwd() + "/figs/learning_rate_comparison.pdf", format="pdf")

plt.show()

print("\nSaved plot to figs/learning_rate_comparison.pdf")
print("Done.")
