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
# 2) Model generator
# =====================================================
def build_model():
    model = models.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model


# =====================================================
# 3) Train with different batch sizes
# =====================================================
batch_sizes = [20, 50, 100, 200]
histories_batches = {}

print("\nTraining models with different batch sizes...\n")

for b in batch_sizes:
    print(f"Training with batch size = {b}")
    model_b = build_model()
    history_b = model_b.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,         # fixed epohs
        batch_size=b,
        verbose=0
    )
    histories_batches[b] = history_b.history


# =====================================================
# 4) Plot results
# =====================================================
plt.figure(figsize=(9,6))

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

for (b, c) in zip(batch_sizes, colors):
    plt.plot(histories_batches[b]['loss'],      color=c, linestyle='-',  label=f"Trening (batch={b})")
    plt.plot(histories_batches[b]['val_loss'], color=c, linestyle='--', label=f"Validacija (batch={b})")

plt.xlabel("Epoka")
plt.ylabel("Napaka (MSE)")
plt.title("Vpliv velikosti batcha na potek učenja", fontsize=14)
plt.grid(True)
plt.ylim(0, 0.002)
plt.legend()
plt.tight_layout()

# Save PDF to figs/
os.makedirs("figs", exist_ok=True)
plt.savefig(os.path.join(cwd, "figs", "batch_comparison.pdf"), format="pdf")

plt.show()

print("\nSaved figure to figs/batch_comparison.pdf")
print("Done.")
