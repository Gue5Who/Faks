import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =====================================================
# SETTINGS
# =====================================================
cwd = os.getcwd()
K_SCALE = 1e-11     # Scale for K11 and K33
os.makedirs("figs", exist_ok=True)

# Find datasets
bc_dirs = sorted(glob.glob(os.path.join(cwd, "DataK13_*")))
if not bc_dirs:
    raise RuntimeError("ERROR: No DataK13_* directories found.")

print("Found boundary-condition sets:")
for d in bc_dirs:
    print("  ", os.path.basename(d))


# =====================================================
# MODEL BUILDER
# =====================================================
def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(2)   # OUTPUT: [K11, K33]
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"]
    )
    return model


# =====================================================
# LOOP OVER DATAK13_* DIRECTORIES
# =====================================================
results = []

for bc_path in bc_dirs:
    bc_name = os.path.basename(bc_path)
    print("\n=====================================================")
    print("Processing:", bc_name)

    # ------------------------------------------
    # LOAD DATA
    # ------------------------------------------
    I_file = os.path.join(bc_path, "intensity505noise0.npy")
    K_file = os.path.join(bc_path, "Kvalues.npy")

    if not os.path.exists(I_file):
        raise FileNotFoundError(f"Missing: {I_file}")
    if not os.path.exists(K_file):
        raise FileNotFoundError(f"Missing: {K_file}")

    I = np.load(I_file).astype("float32")   # shape (N, T)
    Kvals = np.load(K_file).astype("float32")  # shape (N,2) = [K11, K33]

    print("Intensity shape:", I.shape)
    print("Kvalues (K11,K33) shape:", Kvals.shape)

    Y = Kvals.copy()    # unscaled for later
    K11 = Y[:, 0]
    K33 = Y[:, 1]

    # ------------------------------------------
    # SPLIT DATA
    # ------------------------------------------
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        I, Y, test_size=0.2, random_state=42
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=42
    )

    # ------------------------------------------
    # SCALE INPUTS
    # ------------------------------------------
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s   = scaler_X.transform(X_val)
    X_test_s  = scaler_X.transform(X_test)

    # SCALE OUTPUTS (CRITICAL!)
    Y_train_s = Y_train / K_SCALE
    Y_val_s   = Y_val   / K_SCALE
    Y_test_s  = Y_test  / K_SCALE


    # ------------------------------------------
    # TRAIN MODEL
    # ------------------------------------------
    model = build_model(X_train_s.shape[1])
    history = model.fit(
        X_train_s, Y_train_s,
        validation_data=(X_val_s, Y_val_s),
        epochs=80,
        batch_size=128,
        verbose=0
    )

    # ------------------------------------------
    # PREDICT
    # ------------------------------------------
    Y_pred_s = model.predict(X_test_s)
    Y_pred = Y_pred_s * K_SCALE   # back to real units


    # ------------------------------------------
    # ERROR METRICS
    # ------------------------------------------
    err = Y_pred - Y_test
    mae_K11 = np.mean(np.abs(err[:, 0]))
    mae_K33 = np.mean(np.abs(err[:, 1]))

    print(f"MAE K11: {mae_K11:.3e}")
    print(f"MAE K33: {mae_K33:.3e}")

    results.append((bc_name, mae_K11, mae_K33))

    # ------------------------------------------
    # PLOT 2D HISTOGRAMS: K11, K33
    # ------------------------------------------
    for idx, label in enumerate(["K11", "K33"]):
        true_vals = Y_test[:, idx]
        pred_vals = Y_pred[:, idx]

        plt.figure(figsize=(7,7))
        plt.hist2d(true_vals, pred_vals, bins=120, cmap="plasma")
        plt.colorbar(label="Število točk")

        mn = min(true_vals.min(), pred_vals.min())
        mx = max(true_vals.max(), pred_vals.max())
        plt.xlim(mn, mx)
        plt.ylim(mn, mx)
        plt.plot([mn, mx], [mn, mx], "w--", linewidth=2)

        plt.xlabel(f"Pravi {label}")
        plt.ylabel(f"Napovedani {label}")
        plt.title(f"{label} – {bc_name}: Pravi vs napovedani")
        plt.xlim(0, 2e-11)

        plt.tight_layout()
        plt.savefig(f"figs/{bc_name}_{label}_true_vs_pred.pdf", format="pdf")
        plt.close()


# =====================================================
# SUMMARY
# =====================================================
print("\nSUMMARY — MAE per boundary condition:")
for bc_name, mae11, mae33 in results:
    print(f"{bc_name:12s}   K11 MAE = {mae11:.3e}   |   K33 MAE = {mae33:.3e}")

print("\nDone. All results saved in figs/.")