import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1) LOAD DATA
# ============================================================
cwd = os.getcwd()
data_dir = os.path.join(cwd, "DataK")

print("Loading data from:", data_dir)

# no-noise dataset
I_clean  = np.load(os.path.join(data_dir, "intensity505noise0.npy")).astype("float32")
# noise dataset
I_noise  = np.load(os.path.join(data_dir, "intensity505noise100.npy")).astype("float32")

K = np.load(os.path.join(data_dir, "Kvalues.npy")).astype("float32")

print("Shapes:", I_clean.shape, I_noise.shape)

# ============================================================
# 2) SPLIT DATA
# ============================================================
def prepare_dataset(I):
    X_train, X_temp, y_train, y_temp = train_test_split(I, K, test_size=0.2, random_state=42)
    X_val,   X_test, y_val, y_test   = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

Xc_train, Xc_val, Xc_test, yc_train, yc_val, yc_test = prepare_dataset(I_clean)
Xn_train, Xn_val, Xn_test, yn_train, yn_val, yn_test = prepare_dataset(I_noise)

# ============================================================
# 3) SCALING (same scaler for all → fair comparison)
# ============================================================
scaler = StandardScaler()
scaler.fit(Xc_train)        # scale wrt clean data

Xc_train_s = scaler.transform(Xc_train)
Xc_val_s   = scaler.transform(Xc_val)
Xc_test_s  = scaler.transform(Xc_test)

Xn_train_s = scaler.transform(Xn_train)
Xn_val_s   = scaler.transform(Xn_val)
Xn_test_s  = scaler.transform(Xn_test)

# ============================================================
# 4) MODEL GENERATOR
# ============================================================
def build_model():
    model = models.Sequential([
        layers.Input(shape=(Xc_train_s.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse"
    )
    return model


# ============================================================
# 5) TRAINING CASES
# ============================================================

# 5.1 train clean, test clean
m_clean_clean = build_model()
m_clean_clean.fit(Xc_train_s, yc_train, validation_data=(Xc_val_s, yc_val),
                  epochs=80, batch_size=128, verbose=0)
pred_clean_clean = m_clean_clean.predict(Xc_test_s).flatten()

# 5.2 train noise, test noise
m_noise_noise = build_model()
m_noise_noise.fit(Xn_train_s, yn_train, validation_data=(Xn_val_s, yn_val),
                  epochs=80, batch_size=128, verbose=0)
pred_noise_noise = m_noise_noise.predict(Xn_test_s).flatten()

# 5.3 train clean, test noise  (KEY RESULT)
m_clean_noise = build_model()
m_clean_noise.fit(Xc_train_s, yc_train, validation_data=(Xc_val_s, yc_val),
                  epochs=80, batch_size=128, verbose=0)
pred_clean_noise = m_clean_noise.predict(Xn_test_s).flatten()


# ============================================================
# 6) ERROR COMPUTATION
# ============================================================
err_cc = pred_clean_clean - yc_test
err_nn = pred_noise_noise - yn_test
err_cn = pred_clean_noise - yn_test      # clean → noisy (worst)


# ============================================================
# 7) PLOT ERRORS (ALL ON SAME GRAPH)
# ============================================================
plt.figure(figsize=(9,6))
plt.hist(err_cc, bins=80, alpha=0.6, label="Trening brez šuma → Test brez šuma")
plt.hist(err_nn, bins=80, alpha=0.6, label="Trening s šumom → Test s šumom")
plt.hist(err_cn, bins=180, alpha=0.6, label="Trening brez šuma → Test s šumom", color="red")

plt.xlabel("Napaka (K_pred - K_true)")
plt.ylabel("Število primerov")
plt.title("Primerjava napak pri različnih učnih podatkih")
plt.legend()
plt.tight_layout()
plt.xlim(-0.005,0.005)

os.makedirs("figs", exist_ok=True)
plt.savefig("figs/noise_error_hist.pdf", format="pdf")
plt.show()


print("\nDone. Figures saved in figs/")