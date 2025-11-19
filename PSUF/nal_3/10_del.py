import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler


# ============================================================
# SETTINGS
# ============================================================

cwd = os.getcwd()

train_dir = os.path.join(cwd, "DataK13_ph")     # PH geometry (same as experiment)
exp_dir   = os.path.join(cwd, "ExpData")

TARGET_POINTS = 400           # NN input dimension
T_original = 1.95             # original experimental duration (seconds)
T_target = 1.2
K_SCALE = 1e-11               # scaling for K11 and K33
N_ENSEMBLE = 10               # number of networks in ensemble

WINDOW_POINTS = 2000          # how large a segment we cut from the experimental signal
STEP_POINTS   = 1000          # step between intervals

os.makedirs("figs", exist_ok=True)


# ============================================================
# 1) LOAD THEORETICAL TRAINING DATA (PH dataset)
# ============================================================

print("\nLoading training dataset:", train_dir)

I_train = np.load(os.path.join(train_dir, "intensity505noise0.npy")).astype("float32")
K_train = np.load(os.path.join(train_dir, "Kvalues.npy")).astype("float32")

Y_train_s = K_train / K_SCALE   # rescale K11, K33


# ============================================================
# 2) SCALER
# ============================================================

scaler_X = StandardScaler()
scaler_X.fit(I_train)


# ============================================================
# 3) MODEL DEFINITION
# ============================================================

def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(2)   # output = [K11, K33]
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


# ============================================================
# 4) LOAD EXPERIMENTAL DATA
# ============================================================

print("\nLoading experimental intensities:", exp_dir)

exp_files = sorted([f for f in os.listdir(exp_dir) if f.endswith(".npy")])
exp_raw = np.array([np.load(os.path.join(exp_dir, f)).astype("float32") 
                    for f in exp_files])

N_exp, N_orig = exp_raw.shape
print("Experimental signals:", exp_raw.shape)

t_orig = np.linspace(0, T_original, N_orig)
t_new = np.linspace(0, T_target, TARGET_POINTS)


# ============================================================
# 5) CREATE SEGMENTS FROM EXPERIMENTS
# ============================================================

print("\nCreating multiple segments from each experimental signal...")

interval_segments = []

for i in range(N_exp):
    curve = exp_raw[i]

    start = 0
    while start + WINDOW_POINTS <= N_orig:

        segment = curve[start:start + WINDOW_POINTS]

        # Normalize local time from 0 to 1
        seg_t = np.linspace(0, 1, WINDOW_POINTS)
        seg_t_new = np.linspace(0, 1, TARGET_POINTS)

        f = interp1d(seg_t, segment, kind="cubic")
        segment_interp = f(seg_t_new)

        interval_segments.append(segment_interp)

        start += STEP_POINTS

interval_segments = np.array(interval_segments)
print("Total segments generated:", interval_segments.shape)


# ============================================================
# 6) SCALE SEGMENTS
# ============================================================

segments_scaled = scaler_X.transform(interval_segments)


# ============================================================
# 7) TRAIN A SINGLE MODEL AND PREDICT
# ============================================================

print("\nTraining baseline (single) model...")

model_single = build_model(TARGET_POINTS)
model_single.fit(
    scaler_X.transform(I_train),
    Y_train_s,
    epochs=80,
    batch_size=128,
    verbose=0
)

pred_single_s = model_single.predict(segments_scaled)
pred_single = pred_single_s * K_SCALE

K11_single = pred_single[:, 0]
K33_single = pred_single[:, 1]


# ============================================================
# 8) TRAIN ENSEMBLE OF NETWORKS
# ============================================================

print(f"\nTraining ensemble of {N_ENSEMBLE} networks...\n")

ensemble_predictions = []

for i in range(N_ENSEMBLE):
    print(f"Training model {i+1}/{N_ENSEMBLE}...")

    m = build_model(TARGET_POINTS)
    m.fit(
        scaler_X.transform(I_train),
        Y_train_s,
        epochs=80,
        batch_size=128,
        verbose=0
    )

    pred_s = m.predict(segments_scaled)
    pred = pred_s * K_SCALE

    ensemble_predictions.append(pred)

ensemble_predictions = np.array(ensemble_predictions)
ensemble_mean = np.mean(ensemble_predictions, axis=0)

K11_ens = ensemble_mean[:, 0]
K33_ens = ensemble_mean[:, 1]


# ============================================================
# 9) PLOTS
# ============================================================

# ---- K11 ----
plt.figure(figsize=(8,5))
plt.hist(K11_single * 1e12, bins=20, alpha=0.4, label="Single model")
plt.hist(K11_ens * 1e12, bins=20, alpha=0.7, label="Ensemble")
plt.axvline(6.6, color="red", linestyle="--", label="True K11 = 6.6 pN")
plt.xlabel("K11 (pN)")
plt.ylabel("Count")
plt.title("K11 Predictions: Single vs Ensemble")
plt.legend()
plt.tight_layout()
plt.savefig("figs/ensemble_K11_hist.pdf", format="pdf")
plt.show()

# ---- K33 ----
plt.figure(figsize=(8,5))
plt.hist(K33_single * 1e12, bins=20, alpha=0.4, label="Single model")
plt.hist(K33_ens * 1e12, bins=20, alpha=0.7, label="Ensemble")
plt.axvline(9.0, color="red", linestyle="--", label="True K33 = 9.0 pN")
plt.xlabel("K33 (pN)")
plt.ylabel("Count")
plt.title("K33 Predictions: Single vs Ensemble")
plt.legend()
plt.tight_layout()
plt.savefig("figs/ensemble_K33_hist.pdf", format="pdf")
plt.show()


# ============================================================
# 10) PRINT SUMMARY
# ============================================================

print("\n==============================================")
print(" SINGLE MODEL RESULTS:")
print("  Mean K11 =", np.mean(K11_single) * 1e12, "pN")
print("  Mean K33 =", np.mean(K33_single) * 1e12, "pN")
print("----------------------------------------------")
print(" ENSEMBLE AVERAGE RESULTS:")
print("  Mean K11 =", np.mean(K11_ens) * 1e12, "pN")
print("  Mean K33 =", np.mean(K33_ens) * 1e12, "pN")
print("----------------------------------------------")
print(" True K11 = 6.6 pN")
print(" True K33 = 9.0 pN")
print("==============================================\n")

print("All figures saved in figs/")
