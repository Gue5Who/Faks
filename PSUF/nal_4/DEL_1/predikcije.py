import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import keras   # <-- IMPORTANT
keras.config.enable_unsafe_deserialization()   # <-- MAGIC FIX
from keras.saving import register_keras_serializable

@register_keras_serializable()
class LuongAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Wa = None

    def build(self, input_shape):
        units = input_shape[0][-1]
        self.Wa = self.add_weight(
            name="Wa",
            shape=(units, units),
            initializer="glorot_uniform",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs = [query, values]
        query, values = inputs  # query: (B,1,H), values: (B,T,H)

        # score = q * W * V^T
        query_trans = tf.tensordot(query, self.Wa, axes=[[2],[0]])
        score = tf.matmul(query_trans, values, transpose_b=True)

        # softmax attention weights
        weights = tf.nn.softmax(score, axis=-1)

        # weighted sum -> context vector
        context = tf.matmul(weights, values)

        return context, weights


# ================================================================
# Nastavitve datotek
# ================================================================
data_dir = "data_lorentz/"
model_dir = "models_lorenz/"

simple_model_path = model_dir + "lorenz_simple_lstm.keras"
att_model_path    = model_dir + "lorenz_lstm_luong.keras"

scaler_mean  = np.load(data_dir + "lorenz_scaler_mean.npy")
scaler_scale = np.load(data_dir + "lorenz_scaler_scale.npy")

def apply_scaler(x): return (x - scaler_mean) / scaler_scale
def invert_scaler(x): return x * scaler_scale + scaler_mean

os.makedirs("figs", exist_ok=True)

# ================================================================
# Lorenz sistem (RK4) — popolnoma enako kot v tvojem 1. delu
# ================================================================
def lorenz63_rhs(state, sigma=10, r=28, b=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (r - z) - y
    dz = x * y - b * z
    return np.array([dx, dy, dz])

def rk4_step(state, dt):
    k1 = lorenz63_rhs(state)
    k2 = lorenz63_rhs(state + 0.5*dt*k1)
    k3 = lorenz63_rhs(state + 0.5*dt*k2)
    k4 = lorenz63_rhs(state + dt*k3)
    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def integrate_lorenz(x0, t_end=20.0, dt=0.01):
    steps = int(t_end / dt)
    traj = np.zeros((steps, 3))
    state = x0.copy()
    for i in range(steps):
        traj[i] = state
        state = rk4_step(state, dt)
    return traj

# ================================================================
# Iterativna napoved modela
# ================================================================
def iterative_predict(model, true_traj, seq_len=20):
    x_norm = apply_scaler(true_traj)
    seq = x_norm[:seq_len].copy()
    preds_norm = []

    for t in range(seq_len, len(x_norm)):
        inp = seq[np.newaxis, :, :]
        pred = model.predict(inp, verbose=0)[0]
        preds_norm.append(pred)
        seq = np.vstack([seq[1:], pred])

    preds = invert_scaler(np.array(preds_norm))
    return preds

# ================================================================
# Generiraj testno trajektorijo
# ================================================================
dt = 0.01
x0 = np.array([1.0, 1.0, 1.0])
true_traj = integrate_lorenz(x0, t_end=20.0, dt=dt)

# ================================================================
# Naloži modele
# ================================================================
print("Loading models...")
simple_model = keras.models.load_model(simple_model_path)
att_model    = keras.models.load_model(att_model_path, compile=False)

# ================================================================
# Naredi napovedi
# ================================================================
print("Generating predictions...")
pred_simple = iterative_predict(simple_model, true_traj, seq_len=20)
pred_att    = iterative_predict(att_model,    true_traj, seq_len=20)

# Poravnaj true trajektorijo
true_cut = true_traj[20:]

t = np.arange(len(true_cut)) * dt

# ================================================================
# PLOT 1 — X, Y, Z primerjava (true vs simple vs attention)
# ================================================================
labels = ["X", "Y", "Z"]
fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

for i in range(3):
    ax[i].plot(t, true_cut[:, i], label="True", color="black", lw=1.2)
    ax[i].plot(t, pred_simple[:, i], "--", label="Simple LSTM", alpha=0.7)
    ax[i].plot(t, pred_att[:, i], "--", label="Attention LSTM", alpha=0.7)
    ax[i].set_ylabel(labels[i])
    ax[i].grid(alpha=0.3)

ax[-1].set_xlabel("t")
ax[0].legend()
plt.tight_layout()
plt.savefig("figs/lorenz_XYZ_predictions.pdf")
plt.close()
print("Saved figs/lorenz_XYZ_predictions.pdf")

# ================================================================
# PLOT 2 — 3D primerjava trajektorij
# ================================================================
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection="3d")

ax.plot(true_cut[:,0], true_cut[:,1], true_cut[:,2], color="black", lw=0.8, label="True")
ax.plot(pred_simple[:,0], pred_simple[:,1], pred_simple[:,2], "--", label="Simple LSTM")
ax.plot(pred_att[:,0], pred_att[:,1], pred_att[:,2], "--", label="Attention LSTM")

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_title("3D Lorenz – True vs Predictions")
ax.legend()

plt.tight_layout()
plt.savefig("figs/lorenz_3D_predictions.pdf")
plt.close()
print("Saved figs/lorenz_3D_predictions.pdf")