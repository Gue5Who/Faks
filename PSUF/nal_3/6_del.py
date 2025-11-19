import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =====================================================
# 1) LOAD DATA
# =====================================================
cwd = os.getcwd()
data_dir = os.path.join(cwd, "DataK")

print("Loading data from:", data_dir)

I_full = np.load(os.path.join(data_dir, "intensity505noise0.npy")).astype("float32")
K      = np.load(os.path.join(data_dir, "Kvalues.npy")).astype("float32")

print("Full signal shape:", I_full.shape)   # (N, 400)


# =====================================================
# 2) SIMPLE MODEL GENERATOR
# =====================================================
def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='mse'
    )
    return model


# =====================================================
# 3) RESAMPLING FUNCTION
# =====================================================
def resample_signal_matrix(I, new_points):
    N, old_len = I.shape
    x_old = np.linspace(0, 1, old_len)
    x_new = np.linspace(0, 1, new_points)
    I_new = np.zeros((N, new_points), dtype="float32")
    for i in range(N):
        I_new[i, :] = np.interp(x_new, x_old, I[i, :])
    return I_new


# =====================================================
# 4) TRAIN + PLOT FOR MULTIPLE TIME SAMPLINGS
# =====================================================
time_points = [200, 100, 50, 25]     # 200, 100, 50, 25 points
histories = {}

print("\nTraining models for different time resolutions...\n")

for pts in time_points:
    print(f"Resampling to {pts} time points...")

    I_res = resample_signal_matrix(I_full, pts)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(I_res, K, test_size=0.2, random_state=42)
    X_val,   X_test, y_val, y_test   = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    # Train
    model = build_model(pts)
    history = model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=80,
        batch_size=128,
        verbose=0
    )
    histories[pts] = history.history


# =====================================================
# 5) PLOT RESULTS (ALL IN ONE FIG)
# =====================================================
plt.figure(figsize=(9,6))

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

for (pts, c) in zip(time_points, colors):
    plt.plot(histories[pts]['loss'],      color=c, linestyle='-',  label=f"Trening ({pts} točk)")
    plt.plot(histories[pts]['val_loss'], color=c, linestyle='--', label=f"Validacija ({pts} točk)")

plt.xlabel("Epoka")
plt.ylabel("Napaka (MSE)")
plt.title("Vpliv števila časovnih točk na potek učenja")
plt.grid(True)
plt.tight_layout()

os.makedirs("figs", exist_ok=True)
plt.savefig(os.path.join("figs", "time_resolution_comparison.pdf"), format="pdf")
plt.show()


# =====================================================
# 6) SECOND PART — TRAIN ON A TIME WINDOW SUBSET
# =====================================================
print("\nExtracting time window: [0.12s, 1.2s] ...")

# full signal is 400 points from [0, 1.2 s]
T_total = 1.2
N_full = I_full.shape[1]
t_full = np.linspace(0, T_total, N_full)

# pick window from 0.12 s to 1.2 s
t_min = 0.12
t_max = 1.20

mask = (t_full >= t_min)
I_cut = I_full[:, mask]     # reduces to ~360 points

print("Cut signal shape:", I_cut.shape)

# split
X_train, X_temp, y_train, y_temp = train_test_split(I_cut, K, test_size=0.2, random_state=42)
X_val,   X_test, y_val, y_test   = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)

# train
model_cut = build_model(I_cut.shape[1])
history_cut = model_cut.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=80,
    batch_size=128,
    verbose=0
)

# plot
plt.figure(figsize=(9,6))
plt.plot(history_cut.history['loss'], label="Trening")
plt.plot(history_cut.history['val_loss'], '--', label="Validacija")
plt.xlabel("Epoka")
plt.ylabel("Napaka (MSE)")
plt.title("Učenje na časovnem izseku [0.12 s, 1.2 s]")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join("figs", "time_window_training.pdf"), format="pdf")
plt.show()

print("\nDone. Saved all figures in figs/")