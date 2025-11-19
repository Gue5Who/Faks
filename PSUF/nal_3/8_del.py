import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow.keras import layers, models

# =================================================================
# SETTINGS
# =================================================================
cwd = os.getcwd()
exp_dir = os.path.join(cwd, "ExpData")
data_dir = os.path.join(cwd, "DataK13_ph")  # use PH dataset (planar-homeotropic same as experiment)

TARGET_POINTS = 400
T_original = 1.95
T_target = 1.2
K_SCALE = 1e-11

os.makedirs("figs", exist_ok=True)

print("Loading experimental intensities from:", exp_dir)

# =================================================================
# 1) LOAD EXPERIMENTAL INTENSITIES
# =================================================================
exp_files = sorted([f for f in os.listdir(exp_dir) if f.endswith(".npy")])
exp_curves = []

for fname in exp_files:
    arr = np.load(os.path.join(exp_dir, fname)).astype("float32")   # shape (15600,)
    exp_curves.append(arr)

exp_curves = np.array(exp_curves)    # shape (30, 15600)
N_exp, N_orig = exp_curves.shape

print("Loaded experimental curves:", exp_curves.shape)


# =================================================================
# 2) INTERPOLATE EACH EXPERIMENTAL CURVE INTO 400 POINTS OVER 0–1.2 s
# =================================================================
t_orig = np.linspace(0, T_original, N_orig)
t_new = np.linspace(0, T_target, TARGET_POINTS)

exp_resampled = np.zeros((N_exp, TARGET_POINTS), dtype="float32")

for i in range(N_exp):
    f = interp1d(t_orig, exp_curves[i, :], kind="cubic")
    exp_resampled[i, :] = f(t_new)

print("Resampled experimental shape:", exp_resampled.shape)


# =================================================================
# 3) LOAD TRAINING DATA (TO REUSE SCALER)
# =================================================================
# Using PH dataset because experiment PH (planar-homeotropic)
I_train = np.load(os.path.join("DataK13_ph", "intensity505noise0.npy")).astype("float32")
Kvals_train = np.load(os.path.join("DataK13_ph", "Kvalues.npy")).astype("float32")

# Scale K
Y_train_s = Kvals_train / K_SCALE

# Train/val/test split not needed — only scaler is needed
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_X.fit(I_train)  # fit only on theoretical dataset

exp_scaled = scaler_X.transform(exp_resampled)


# =================================================================
# 4) BUILD BEST MODEL (same as task7)
# =================================================================
def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(2)   # outputs K11, K33
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse"
    )
    return model


# =================================================================
# 5) TRAIN MODEL ON THEORETICAL DATA
# =================================================================
model = build_model(I_train.shape[1])
model.fit(
    scaler_X.transform(I_train), Y_train_s,
    epochs=80,
    batch_size=128,
    verbose=1
)

# =================================================================
# 6) PREDICT ON EXPERIMENTAL DATA
# =================================================================
pred_exp_s = model.predict(exp_scaled)
pred_exp = pred_exp_s * K_SCALE   # back to physical units

K11_pred = pred_exp[:, 0]
K33_pred = pred_exp[:, 1]

# True known experimental values for 5CB at RT
K11_true = 6.6e-12     # pN → SI? 6.6 pN = 6.6e-12 N
K33_true = 9.0e-12


# =================================================================
# 7) PLOTS
# =================================================================

# ---- Histogram for K11 ----
plt.figure(figsize=(8,5))
plt.hist(K11_pred*1e12, bins=15, alpha=0.8, color="blue")
plt.axvline(6.6, color="red", linestyle="--", label="K11 pravi (6.6 pN)")
plt.xlabel("Napovedani K11 (pN)")
plt.ylabel("Število meritev")
plt.title("Porazdelitev napovedanih K11 iz eksperimenta")
plt.legend()
plt.tight_layout()
plt.savefig("figs/exp_K11_hist.pdf", format="pdf")
plt.show()

# ---- Histogram for K33 ----
plt.figure(figsize=(8,5))
plt.hist(K33_pred*1e12, bins=15, alpha=0.8, color="green")
plt.axvline(9.0, color="red", linestyle="--", label="K33 pravi (9.0 pN)")
plt.xlabel("Napovedani K33 (pN)")
plt.ylabel("Število meritev")
plt.title("Porazdelitev napovedanih K33 iz eksperimenta")
plt.legend()
plt.tight_layout()
plt.savefig("figs/exp_K33_hist.pdf", format="pdf")
plt.show()

# Print results
print("\n==============================================")
print("EXPERIMENTAL PREDICTION RESULTS")
print("Mean predicted K11 =", np.mean(K11_pred)*1e12, "pN")
print("Mean predicted K33 =", np.mean(K33_pred)*1e12, "pN")
print("True K11 =", 6.6, "pN")
print("True K33 =", 9.0, "pN")
print("==============================================\n")

print("Done. Results saved in figs/")
