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
# 2) Hyperparameter: number of neurons per layer
# =====================================================
neuron_counts = [32, 64, 128, 256]
histories_neurons = {}

print("\nTraining models with different neuron counts...\n")


# =====================================================
# 3) Model builder helper
# =====================================================
def build_model_neurons(n_neurons):
    model = models.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(n_neurons, activation='relu'),
        layers.Dense(n_neurons, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model


# =====================================================
# 4) Train models for each neuron count
# =====================================================
for n in neuron_counts:
    print(f"Training with {n} neurons per layer")
    model_n = build_model_neurons(n)
    history_n = model_n.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=120,
        batch_size=128,
        verbose=0
    )
    histories_neurons[n] = history_n.history


# =====================================================
# 5) Plot results
# =====================================================
plt.figure(figsize=(9,6))

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

for (n, c) in zip(neuron_counts, colors):
    plt.plot(histories_neurons[n]['loss'],      color=c, linestyle='-',  label=f"Trening – {n} nevronov")
    plt.plot(histories_neurons[n]['val_loss'], color=c, linestyle='--', label=f"Validacija – {n} nevronov")

plt.xlabel("Epoka")
plt.ylabel("Napaka (MSE)")
plt.title("Vpliv števila nevronov na potek učenja", fontsize=14)
plt.grid(True)
plt.ylim(0, 0.002)
plt.legend()
plt.tight_layout()

# Save PDF to figs/
os.makedirs("figs", exist_ok=True)
plt.savefig(os.getcwd() + "/figs/neuron_count_comparison.pdf", format="pdf")

plt.show()

print("\nSaved plot to figs/neuron_count_comparison.pdf")
print("Done.")
