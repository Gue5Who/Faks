import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from keras.saving import register_keras_serializable
from train_lorenz_models_3 import LuongAttention


# --------------------------------------------------------
# Load your scaler from Part 1
# --------------------------------------------------------

import keras

keras.config.enable_unsafe_deserialization()

att_model = keras.models.load_model(
    "models_lorenz/lorenz_lstm_luong.keras",
    compile=False
)



data_dir = 'data_lorentz/'

scaler_mean  = np.load(data_dir + "lorenz_scaler_mean.npy")
scaler_scale = np.load(data_dir + "lorenz_scaler_scale.npy")

def apply_scaler(x):
    return (x - scaler_mean) / scaler_scale

def invert_scaler(x):
    return x * scaler_scale + scaler_mean


@register_keras_serializable()
class LuongAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Wa = None

    def build(self, input_shape):
        units = input_shape[0][-1]
        self.Wa = self.add_weight(
            name="Wa", shape=(units, units),
            initializer="glorot_uniform",
            trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        query, values = inputs
        query_transformed = tf.tensordot(query, self.Wa, axes=[[2],[0]])
        score = tf.matmul(query_transformed, values, transpose_b=True)
        weights = tf.nn.softmax(score, axis=-1)
        context = tf.matmul(weights, values)
        return context, weights

# --------------------------------------------------------
# Lorenz integrator (same as in Part 1)
# --------------------------------------------------------
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

def integrate_lorenz(x0, t_end=10.0, dt=0.01):
    steps = int(t_end / dt)
    traj = np.zeros((steps, 3))
    state = x0.copy()
    for i in range(steps):
        state = rk4_step(state, dt)
        traj[i] = state
    return traj

# --------------------------------------------------------
# Load trained models
# --------------------------------------------------------
simple_model = keras.models.load_model("models_lorenz/lorenz_simple_lstm.keras")
att_model    = keras.models.load_model("models_lorenz/lorenz_lstm_luong.keras", compile=False)

INPUT_SEQ_LEN = 20
DT = 0.01
T_END = 10.0
STEPS = int(T_END / DT)
N_TEST = 200

# --------------------------------------------------------
# Generate random initial conditions
# --------------------------------------------------------
ics = np.random.uniform(-20, 20, size=(N_TEST, 3))

# --------------------------------------------------------
# Function: iterative NN prediction
# --------------------------------------------------------
def iterative_prediction(model, x0_true, input_seq_len):
    """
    x0_true: full TRUE trajectory (unnormalized)
    """
    # Normalize true trajectory
    x_norm = apply_scaler(x0_true)

    # Seed the model with the first k points
    seq = x_norm[:input_seq_len].copy()
    preds_norm = []

    for t in range(input_seq_len, STEPS):
        inp = seq[np.newaxis, :, :]        # shape (1, k, 3)
        pred = model.predict(inp, verbose=0)[0]   # shape (3,)
        preds_norm.append(pred)

        # slide window
        seq = np.vstack([seq[1:], pred])

    preds_norm = np.array(preds_norm)
    preds = invert_scaler(preds_norm)
    return preds

# --------------------------------------------------------
# Compute predictions and errors
# --------------------------------------------------------
errors_simple = []
errors_att = []

# store one example trajectory index
example_idx = 0
example_true = None
example_pred_simple = None
example_pred_att = None

for i in range(N_TEST):
    print(f"Evaluating IC {i+1}/{N_TEST} ...")
    # integrate true trajectory
    x0 = ics[i]
    true_traj = integrate_lorenz(x0, t_end=T_END, dt=DT)

    # iterative NN prediction for both models
    pred_simple = iterative_prediction(simple_model, true_traj, INPUT_SEQ_LEN)
    pred_att    = iterative_prediction(att_model, true_traj, INPUT_SEQ_LEN)

    # truncate true to match prediction length
    true_cut = true_traj[INPUT_SEQ_LEN:]

    # compute per-step Euclidean error
    err_simple = np.linalg.norm(pred_simple - true_cut, axis=1)
    err_att    = np.linalg.norm(pred_att    - true_cut, axis=1)

    errors_simple.append(err_simple)
    errors_att.append(err_att)

    if i == example_idx:
        example_true = true_cut
        example_pred_simple = pred_simple
        example_pred_att = pred_att

errors_simple = np.array(errors_simple)   # shape (N_TEST, STEPS-k)
errors_att = np.array(errors_att)

mean_err_simple = errors_simple.mean(axis=0)
mean_err_att    = errors_att.mean(axis=0)

time_axis = np.arange(len(mean_err_simple)) * DT

# --------------------------------------------------------
# Plot 1: Example true vs predicted trajectories
# --------------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(time_axis, example_true[:,0], label="True X", color="black")
plt.plot(time_axis, example_pred_simple[:,0], "--", label="Simple LSTM", alpha=0.7)
plt.plot(time_axis, example_pred_att[:,0], "--", label="Attention LSTM", alpha=0.7)
plt.xlabel("t")
plt.ylabel("X")
plt.title("Trajectory comparison (X component)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("traj_compare_X.png")
plt.close()

# --------------------------------------------------------
# Plot 2: Error growth E(t)
# --------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(time_axis, mean_err_simple, label="Simple LSTM")
plt.plot(time_axis, mean_err_att, label="Attention LSTM")
plt.yscale("log")
plt.xlabel("t")
plt.ylabel("Mean error ||pred - true||")
plt.title("Error growth (chaotic divergence)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("error_growth.png")
plt.close()

print("Evaluation finished. Saved: traj_compare_X.png, error_growth.png")