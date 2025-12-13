import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.saving import register_keras_serializable

# adjust this import if your function has a different name
from prep_sec_2 import create_sequences   # or split_sequences, etc.

# ============================================================
# 0. General settings
# ============================================================

INPUT_SEQ_LEN = 20         # k in the assignment; change & re-train to study effect
UNITS_RNN      = 64        # hidden size for simple LSTM
UNITS_ATT      = 64        # hidden size for attention LSTM
BATCH_SIZE     = 256
EPOCHS         = 40

MODEL_DIR = "models_lorenz"
os.makedirs(MODEL_DIR, exist_ok=True)

data_dir = 'data_lorentz/'

# ============================================================
# 1. Load data and prepare sequences (many-to-one)
# ============================================================

train_data = np.load(data_dir + "lorenz_train.npy")   # shape (N_train, 3)
val_data   = np.load(data_dir + "lorenz_val.npy")     # shape (N_val, 3)

# create supervised datasets:
#   [X(t-k+1), ..., X(t)] -> X(t+Δt)
X_train, y_train = create_sequences(train_data, input_seq_len=INPUT_SEQ_LEN)
X_val,   y_val   = create_sequences(val_data,   input_seq_len=INPUT_SEQ_LEN)

print("X_train:", X_train.shape)   # (n_samples, INPUT_SEQ_LEN, 3)
print("y_train:", y_train.shape)   # (n_samples, 3)


# ============================================================
# 2. Model A1: Simple LSTM (many-to-one)
#    (You can switch to SimpleRNN if you prefer)
# ============================================================

def build_simple_lstm(input_seq_len, units=64):
    model = keras.Sequential(
        [
            layers.Input(shape=(input_seq_len, 3)),
            layers.LSTM(units, return_sequences=False),
            layers.Dense(3, activation="linear"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="mse",
        metrics=["mae", "mse"],
    )
    return model


simple_model = build_simple_lstm(INPUT_SEQ_LEN, UNITS_RNN)
simple_model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
    ),
]

history_simple = simple_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

simple_model.save(os.path.join(MODEL_DIR, "lorenz_simple_lstm.keras"))
np.save(os.path.join(MODEL_DIR, "lorenz_simple_lstm_hist.npy"),
        history_simple.history)


# ============================================================
# 3. Luong Attention layer
# ============================================================

@register_keras_serializable()
class LuongAttention(layers.Layer):
    """
    Luong attention (general / dot style).
    query:  (batch, T_dec, units)   - decoder outputs
    values: (batch, T_enc, units)   - encoder outputs
    returns:
        context: (batch, T_dec, units)
        weights: (batch, T_dec, T_enc)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # "general" style: W_a matrix
        self.Wa = None

    def build(self, input_shape):
        # input_shape = [query_shape, values_shape]
        units = input_shape[0][-1]
        self.Wa = self.add_weight(
            name="Wa",
            shape=(units, units),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        query, values = inputs  # query: decoder outputs, values: encoder outputs

        # (batch, T_dec, units) @ (units, units) -> (batch, T_dec, units)
        query_transformed = tf.tensordot(query, self.Wa, axes=[[2], [0]])

        # score = query_transformed * values^T  -> (batch, T_dec, T_enc)
        score = tf.matmul(query_transformed, values, transpose_b=True)

        # attention weights over encoder time steps
        weights = tf.nn.softmax(score, axis=-1)

        # context = sum_i alpha_i * h_i  -> (batch, T_dec, units)
        context = tf.matmul(weights, values)

        return context, weights


# ============================================================
# 4. Model A2: Encoder–decoder LSTM with Luong attention
#    (many-to-one, output length = 1)
# ============================================================

def build_lstm_luong(input_seq_len, units=64, output_seq_len=1):
    """
    Encoder–decoder LSTM with Luong attention.
    We use many-to-one setting by choosing output_seq_len = 1.
    """
    # ----- Encoder -----
    encoder_inputs = keras.Input(shape=(input_seq_len, 3), name="encoder_inputs")
    encoder_lstm = layers.LSTM(
        units,
        return_sequences=True,
        return_state=True,
        name="encoder_lstm",
    )
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

    # ----- Decoder -----
    # Start decoder from last encoder state.
    # For many-to-one we only need a single time step (output_seq_len = 1).
    decoder_inputs = layers.RepeatVector(output_seq_len)(state_h)  # (batch, 1, units)

    decoder_lstm = layers.LSTM(
        units,
        return_sequences=True,
        return_state=False,
        name="decoder_lstm",
    )
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])

    # ----- Luong attention -----
    attention_layer = LuongAttention(name="luong_attention")
    context, attn_weights = attention_layer([decoder_outputs, encoder_outputs])

    # Combine context and decoder output
    decoder_combined = layers.Concatenate(axis=-1)([decoder_outputs, context])

    # Final dense layers to get 3D output
    # decoder_combined shape: (batch, 1, 2*units)
    dense_out = layers.TimeDistributed(
        layers.Dense(32, activation="tanh")
    )(decoder_combined)
    dense_out = layers.TimeDistributed(
        layers.Dense(3, activation="linear")
    )(dense_out)

    # For many-to-one we can squeeze the time dimension (1)
    final_output = layers.Lambda(lambda x: x[:, 0, :], name="output_squeeze")(dense_out)

    model = keras.Model(
        inputs=encoder_inputs,
        outputs=final_output,
        name="lstm_luong_many_to_one",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="mse",
        metrics=["mae", "mse"],
    )

    return model


att_model = build_lstm_luong(INPUT_SEQ_LEN, UNITS_ATT, output_seq_len=1)
att_model.summary()

history_att = att_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

att_model.save(os.path.join(MODEL_DIR, "lorenz_lstm_luong.keras"))
np.save(os.path.join(MODEL_DIR, "lorenz_lstm_luong_hist.npy"),
        history_att.history)

print("Training finished. Models saved in:", MODEL_DIR)