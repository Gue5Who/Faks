import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


# ======================================================
# 1) LOAD DATA
# ======================================================
I = np.load("DataK/intensity505noise0.npy")       # shape (N, 400)
K = np.load("DataK/Kvalues.npy").astype("float32")

print("Loaded:")
print("  I shape:", I.shape)
print("  K shape:", K.shape)


# ======================================================
# 2) TRAIN/VAL/TEST SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    I, K, test_size=0.10, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.10, random_state=42
)


# ======================================================
# 3) SCALING OF INPUTS AND OUTPUTS
# ======================================================
# ⚠ Extremely important: scale the intensity curves
scaler_X = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_val_s   = scaler_X.transform(X_val)
X_test_s  = scaler_X.transform(X_test)

# ⚠ Also scale K because values are ~1e-11
K_scale = 1e-11
y_train_s = y_train / K_scale
y_val_s   = y_val   / K_scale
y_test_s  = y_test  / K_scale


# ======================================================
# 4) BUILD MODEL
# ======================================================
def build_model():
    model = models.Sequential([
        layers.Input(shape=(X_train_s.shape[1],)),
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


model = build_model()
model.summary()


# ======================================================
# 5) TRAIN MODEL
# ======================================================
es = tf.keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train_s, y_train_s,
    validation_data=(X_val_s, y_val_s),
    epochs=100,
    batch_size=128,
    verbose=1,
    callbacks=[es]
)


# ======================================================
# 6) PREDICT
# ======================================================
y_pred_scaled = model.predict(X_test_s).flatten()
y_pred = y_pred_scaled * K_scale   # unscale back to real K


# ======================================================
# 7) PLOT TRUE vs PREDICTED — 2D histogram
# ======================================================
plt.figure(figsize=(7,7))

h = plt.hist2d(y_test, y_pred, bins=120, cmap="plasma")
plt.colorbar(label="Število točk")

# same scale for x and y
minK = y_test.min()
maxK = y_test.max()
plt.xlim(minK, maxK)
plt.ylim(minK, maxK)

# diagonal line
plt.plot([minK, maxK], [minK, maxK], 'w--', linewidth=2)

plt.xlabel("Pravi K", fontsize=12)
plt.ylabel("Napovedani K", fontsize=12)
plt.title("Primerjava pravih in napovedanih vrednosti K", fontsize=14)

plt.grid(False)
plt.tight_layout()
plt.savefig(os.getcwd() + "/figs/K_pred_vs_true_heatmap.pdf", format = 'pdf')
#plt.show()


# ======================================================
# 8) FIND SYSTEMATIC OUTLIERS
# ======================================================
errors = y_pred - y_test
abs_errors = np.abs(errors)

# take worst 1%
threshold = np.percentile(abs_errors, 99)
bad_idx = np.where(abs_errors >= threshold)[0]

print("Število najhujših odstopanj:", len(bad_idx))


# ======================================================
# 9) PLOT WORST INTENSITY CURVES
# ======================================================
plt.figure(figsize=(8,5))

for i, idx in enumerate(bad_idx[:5]):
    plt.plot(X_test[idx],
             label=f"K_pravi={y_test[idx]:.3g}, K_pred={y_pred[idx]:.3g}")

plt.title("Intenzitetne krivulje z največjimi odstopanji", fontsize=14)
plt.xlabel("Časovni indeks", fontsize=12)
plt.ylabel("I(t)", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlim(0,200)
plt.savefig(os.getcwd() + "/figs/K_outlier_curves.pdf", format = 'pdf')
#plt.show()

print("Done.")