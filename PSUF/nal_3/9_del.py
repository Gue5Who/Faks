import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler

# ============================================================
# SETTINGS
# ============================================================
cwd = os.getcwd()
exp_dir = os.path.join(cwd, "ExpData")
train_dir = os.path.join(cwd, "DataK13_ph")     # PH: planar–homeotropic (same as experiment)

WINDOW_POINTS = 2000        # dolžina intervala, ki ga vzamemo iz 15600 originalnih točk
STEP_POINTS   = 1000        # zamik začetka intervala (manj = več intervalov)
TARGET_POINTS = 400         # mreža uporablja 400 točk
T_original = 1.95
T_target = 1.2
K_SCALE = 1e-11

os.makedirs("figs", exist_ok=True)

# ============================================================
# 1) Load experimental data (30× 15600)
# ============================================================
exp_files = sorted([f for f in os.listdir(exp_dir) if f.endswith(".npy")])
exp_raw = np.array([np.load(os.path.join(exp_dir, f)).astype("float32") for f in exp_files])

N_exp, N_orig = exp_raw.shape
print("Loaded experimental data:", exp_raw.shape)

# ============================================================
# 2) Load training data for scaler + model
# ============================================================
I_train = np.load(os.path.join(train_dir, "intensity505noise0.npy")).astype("float32")
K_train = np.load(os.path.join(train_dir, "Kvalues.npy")).astype("float32")

# Scale K-values
Y_train_s = K_train / K_SCALE

# Fit scaler
scaler_X = StandardScaler()
scaler_X.fit(I_train)

# Build model (same as task 7)
def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(2)   # [K11, K33]
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model

model = build_model(I_train.shape[1])
model.fit(scaler_X.transform(I_train), Y_train_s, epochs=80, batch_size=128, verbose=1)

# ============================================================
# 3) Generate multiple intervals from each curve
# ============================================================
t_orig = np.linspace(0, T_original, N_orig)
t_new = np.linspace(0, T_target, TARGET_POINTS)

interval_segments = []

for i in range(N_exp):
    curve = exp_raw[i]

    start = 0
    while start + WINDOW_POINTS <= N_orig:
        seg = curve[start:start + WINDOW_POINTS]

        # interpolate each segment to 400 points
        # Normalize each segment to local time 0–1
        seg_t = np.linspace(0, 1, WINDOW_POINTS)
        seg_t_new = np.linspace(0, 1, TARGET_POINTS)

        f = interp1d(seg_t, seg, kind="cubic")
        seg_interp = f(seg_t_new)

        interval_segments.append(seg_interp)
        start += STEP_POINTS

interval_segments = np.array(interval_segments)
print("Generated segments:", interval_segments.shape)

# ============================================================
# 4) Predict K11 and K33 for all segments
# ============================================================
segments_scaled = scaler_X.transform(interval_segments)
pred_s = model.predict(segments_scaled)
pred = pred_s * K_SCALE

K11_pred = pred[:, 0]
K33_pred = pred[:, 1]

# true values at room temperature (5CB)
K11_true = 6.6
K33_true = 9.0

# ============================================================
# 5) Plots
# ============================================================

# ---- Histogram K11 ----
plt.figure(figsize=(8, 5))
plt.hist(K11_pred * 1e12, bins=20, color="blue", alpha=0.8)
plt.axvline(K11_true, color="red", linestyle="--", label="Pravi K11 = 6.6 pN")
plt.xlabel("Napovedani K11 (pN)")
plt.ylabel("Število intervalov")
plt.title("Porazdelitev napovedanih K11 iz več intervalov")
plt.legend()
plt.tight_layout()
plt.savefig("figs/exp_intervals_K11_hist.pdf", format="pdf")
plt.show()

# ---- Histogram K33 ----
plt.figure(figsize=(8, 5))
plt.hist(K33_pred * 1e12, bins=20, color="green", alpha=0.8)
plt.axvline(K33_true, color="red", linestyle="--", label="Pravi K33 = 9.0 pN")
plt.xlabel("Napovedani K33 (pN)")
plt.ylabel("Število intervalov")
plt.title("Porazdelitev napovedanih K33 iz več intervalov")
plt.legend()
plt.tight_layout()
plt.savefig("figs/exp_intervals_K33_hist.pdf", format="pdf")
plt.show()

print("\n=================================================")
print(" Mean predicted K11 =", np.mean(K11_pred) * 1e12, "pN")
print(" Mean predicted K33 =", np.mean(K33_pred) * 1e12, "pN")
print(" True K11 =", K11_true, "pN")
print(" True K33 =", K33_true, "pN")
print("=================================================\n")

print("Done. Figures saved in figs/")