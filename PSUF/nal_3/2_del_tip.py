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
data_dir = os.path.join(os.getcwd(), "DataK")

I = np.load(os.path.join(data_dir, "intensity505noise0.npy")).astype("float32")
K = np.load(os.path.join(data_dir, "Kvalues.npy")).astype("float32")

# Train / val / test split
X_train, X_temp, y_train, y_temp = train_test_split(I, K, test_size=0.2, random_state=42)
X_val,   X_test, y_val, y_test   = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalize X
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)


# =====================================================
# 2) Model generator (only activation changes)
# =====================================================
def build_model(activation):
    model = models.Sequential([
        layers.Input(shape=(X_train_s.shape[1],)),
        layers.Dense(128, activation=activation),
        layers.Dense(64, activation=activation),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return model


# =====================================================
# 3) Train with different activation functions
# =====================================================
activations = ['relu', 'tanh', 'sigmoid', 'linear']
histories = {}

for act in activations:
    print(f"Training with activation '{act}'")
    model = build_model(act)
    history = model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=80,
        batch_size=200,
        verbose=0
    )
    histories[act] = history.history


# =====================================================
# 4) Plot learning curves
# =====================================================
plt.figure(figsize=(9,6))

for act in activations:
    plt.plot(histories[act]['loss'],      label=f"Trening – {act}")
    plt.plot(histories[act]['val_loss'], '--', label=f"Validacija – {act}")

plt.title("Vpliv aktivacijske funkcije na potek učenja nevronske mreže", fontsize=13)
plt.xlabel("Epoka")
plt.ylabel("Napaka (MSE)")
plt.grid(True)
plt.legend()
plt.ylim(0, 0.002)
plt.tight_layout()

# Save to figs/ as PDF
os.makedirs("figs", exist_ok=True)
plt.savefig(os.getcwd() + "/figs/activation_function_comparison.pdf", format="pdf")
plt.show()