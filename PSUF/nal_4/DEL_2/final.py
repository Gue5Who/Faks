#!/usr/bin/env python3
"""
PSUF naloga 4, del 2: CNN za Z500 na 4° mreži, 1-dnevni korak.

- Podatki: data/Z500/Z500_4/Z500_4_YYYY.nc
  VARIABLES: time, lon, lat, plev, z
  DIMENSIONS: time, lon, lat, plev

- Mreža: (nlat, nlon) ~ (45, 90)
- Nivo: plev = 50000 Pa (500 hPa)
- Vhod/izhodi:  Z500(t) -> Z500(t+1 dan)
- Latitudinalna standardizacija (mean/std po (čas, lon) za vsako lat)
- Parametriziran CNN kodirnik–dekodirnik
"""

import os
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, UpSampling2D,
    BatchNormalization, Lambda
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# ============================================================
# KONFIGURACIJA
# ============================================================

# Lokacija podatkov (relativno na mapo, iz katere poganjaš skripto)
DATA_DIR = "data/Z500/Z500_4"
FILE_TEMPLATE = "Z500_4_{year}.nc"

# RAZDELITEV LET (prosto prilagodljivo)
ALL_YEARS = list(range(1940, 2025))  # 1940–2024
TRAIN_YEARS = list(range(1940, 2005))  # 1940–2004
VAL_YEARS   = list(range(2005, 2015))  # 2005–2014
TEST_YEARS  = list(range(2015, 2025))  # 2015–2024

BATCH_SIZE = 128
EPOCHS     = 50

FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# BRANJE PODATKOV
# ============================================================



def load_year_z500(year):
    """
    Generalno branje Z500, podpira naslednje oblike:
      (time, lat, lon)
      (time, 1, lat, lon)
      (1, time, lat, lon)
    """

    path = os.path.join(DATA_DIR, FILE_TEMPLATE.format(year=year))
    ds = Dataset(path, "r")

    # lat
    for cand in ["lat", "latitude", "nav_lat"]:
        if cand in ds.variables:
            lats = ds.variables[cand][:]
            break

    # lon
    for cand in ["lon", "longitude", "nav_lon"]:
        if cand in ds.variables:
            lons = ds.variables[cand][:]
            break

    # Z candidate
    for cand in ["z", "Z", "Z500", "var129", "hgt"]:
        if cand in ds.variables:
            z_all = ds.variables[cand][:]
            zname = cand
            break
    else:
        # fallback if unknown
        for v in ds.variables:
            shp = ds.variables[v].shape
            if len(shp) >= 3 and shp[-1] == len(lons) and shp[-2] == len(lats):
                z_all = ds.variables[v][:]
                zname = v
                print(f"⚠ Uporabljam '{v}' kot Z500 kandidat.")
                break

    # Normalize shapes
    if z_all.ndim == 3:
        z = z_all                        # (time, lat, lon)

    elif z_all.ndim == 4:
        if z_all.shape[1] == 1:          # (time, 1, lat, lon)
            z = z_all[:, 0, :, :]
        elif z_all.shape[0] == 1:        # (1, time, lat, lon)
            z = z_all[0, :, :, :]
        else:
            raise RuntimeError(f"Ne morem interpretirati oblike {z_all.shape} spremenljivke {zname}")

    else:
        raise RuntimeError(f"Neznana oblika Z: {z_all.shape}")

    ds.close()
    return z, lats, lons


def stack_years(years):
    all_data = []
    lat = lon = None

    for y in years:
        z, lat_, lon_ = load_year_z500(y)

        if lat is None:
            lat = lat_
            lon = lon_

        all_data.append(z)

    out = np.concatenate(all_data, axis=0)   # [Ndays, lat, lon]
    return out, lat, lon

# ============================================================
# LATITUDINALNA STANDARDIZACIJA
# ============================================================

def latitudinal_standardization(train_data, *other_data):
    """
    Latitudinalna standardizacija:
        X -> (X - mean_lat) / std_lat

    mean_lat in std_lat sta izračunana po dimenzijah (čas, longituda)
    posebej za vsako geografsko širino.

    train_data : [N, nlat, nlon]
    other_data : dodatni seti, ki jih standardiziramo z istimi mean/std

    Vrne:
        train_std, [others_std...], mean_lat, std_lat
    """
    # [1, nlat, 1]
    mean_lat = train_data.mean(axis=(0, 2), keepdims=True)
    std_lat  = train_data.std(axis=(0, 2), keepdims=True)
    std_lat[std_lat == 0] = 1.0  # zaščita pred deljenjem z 0

    def transform(x):
        return (x - mean_lat) / std_lat

    train_std = transform(train_data)
    others_std = [transform(o) for o in other_data]

    return train_std, others_std, mean_lat, std_lat


# ============================================================
# PERIODIČNI PADDING
# ============================================================

def periodic_pad_lambda(kernel_size):
    """
    Vrne Keras Lambda plast, ki implementira periodični rob
    v smeri longituda + "odsevan" z zavitjem v smeri širine.

    kernel_size : (k_lat, k_lon), liha števila.
    """

    k_lat, k_lon = kernel_size

    def pad_fn(x):
        # x: [batch, nlat, nlon, channels]
        north_south_pad = k_lat // 2
        east_west_pad   = k_lon // 2

        arr = x

        # NORTH & SOUTH
        if north_south_pad > 0:
            top = tf.reverse(arr[:, 0:north_south_pad, :, :], axis=[1])
            top = tf.roll(top, shift=int(top.shape[2] // 2), axis=2)

            bottom = tf.reverse(arr[:, -north_south_pad:, :, :], axis=[1])
            bottom = tf.roll(bottom, shift=int(bottom.shape[2] // 2), axis=2)

            arr = tf.concat([top, arr, bottom], axis=1)

        # EAST & WEST (periodika po longitudi)
        if east_west_pad > 0:
            left  = arr[:, :, 0:east_west_pad, :]
            right = arr[:, :, -east_west_pad:, :]
            arr = tf.concat([right, arr, left], axis=2)

        return arr

    return Lambda(pad_fn)


# ============================================================
# CNN MODEL
# ============================================================

def create_cnn_model(
    input_shape,
    filters=(2, 4, 8),
    kernel_sizes=((5, 5), (3, 3), (3, 3)),
    activation="relu",
    use_periodic_padding=True,
    use_batchnorm=False,
    extra_conv_block=False,
):
    """
    Ustvari konvolucijski kodirnik-dekodirnik v slogu create_model_CNN iz navodil,
    z možnostjo:
      - poljubnega števila filtrov (filters)
      - poljubnih dimenzij jeder (kernel_sizes)
      - izbire aktivacijske funkcije (activation)
      - dodatnega conv+pool bloka (extra_conv_block)
      - izbire periodic/zero paddinga v prvi plasti (use_periodic_padding)

    input_shape: (nlat, nlon, 1)
    """
    assert len(filters) >= 3, "filters naj ima vsaj tri elemente!"
    assert len(kernel_sizes) >= 3, "kernel_sizes naj ima vsaj tri elemente!"

    k1, k2, k3 = kernel_sizes[:3]
    f1, f2, f3 = filters[:3]

    inp = Input(shape=input_shape)

    # ------------ Encoder ------------

    # 1. konvolucijska plast
    x = inp
    if use_periodic_padding:
        x = periodic_pad_lambda(k1)(x)
        padding1 = "valid"   # ker smo ročno obložili
    else:
        padding1 = "same"    # klasičen zero padding

    x = Conv2D(f1, k1, activation=activation, padding=padding1)(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3))(x)

    # (opcijsko) dodatni conv+pool blok
    if extra_conv_block:
        x = Conv2D(f1, k1, activation=activation, padding="same")(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)  # pazi, da dimenzije ostanejo kompatibilne

    # 2. konvolucijska plast
    x = Conv2D(f2, k2, activation=activation, padding="same")(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3))(x)

    # latentna plast
    encoded = Conv2D(f3, k3, activation=activation, padding="same", name="latent")(x)
    if use_batchnorm:
        encoded = BatchNormalization()(encoded)

    # ------------ Decoder ------------

    x = UpSampling2D((3, 3))(encoded)
    x = Conv2D(f2, k2, activation=activation, padding="same")(x)
    if use_batchnorm:
        x = BatchNormalization()(x)

    if extra_conv_block:
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(f1, k1, activation=activation, padding="same")(x)
        if use_batchnorm:
            x = BatchNormalization()(x)

    x = UpSampling2D((3, 3))(x)
    x = Conv2D(f1, k1, activation=activation, padding="same")(x)
    if use_batchnorm:
        x = BatchNormalization()(x)

    # zadnja plast – linearna napoved Z500
    out = Conv2D(1, (3, 3), activation="linear", padding="same")(x)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mean_squared_error",
        metrics=["mae", "mse"],
    )
    model.summary()
    return model


# ============================================================
# PRIPRAVA PODATKOV
# ============================================================

def to_pairs(arr_std):
    """
    Vzame latitudinalno standardizirane podatke arr_std [T, nlat, nlon]
    in vrne:
      X(t)  = arr_std[:-1]
      Y(t)  = arr_std[1:]
    z dodano kanalsko dimenzijo.
    """
    x = arr_std[:-1]
    y = arr_std[1:]
    x = x[..., np.newaxis]
    y = y[..., np.newaxis]
    return x, y


def prepare_datasets():
    """
    Naloži train/val/test, naredi latitudinalno standardizacijo in
    vrne:
      x_train, y_train
      x_val,   y_val
      x_test,  y_test
      train_std, val_std, test_std
      lat, lon, mean_lat, std_lat
    """
    train_raw, lat, lon = stack_years(TRAIN_YEARS)
    val_raw,   _,   _   = stack_years(VAL_YEARS)
    test_raw,  _,   _   = stack_years(TEST_YEARS)

    train_std, [val_std, test_std], mean_lat, std_lat = latitudinal_standardization(
        train_raw, val_raw, test_raw
    )

    x_train, y_train = to_pairs(train_std)
    x_val,   y_val   = to_pairs(val_std)
    x_test,  y_test  = to_pairs(test_std)

    print("OBLIKE PODATKOV:")
    print("  x_train:", x_train.shape)
    print("  y_train:", y_train.shape)
    print("  x_val:  ", x_val.shape)
    print("  x_test: ", x_test.shape)

    return (x_train, y_train,
            x_val,   y_val,
            x_test,  y_test,
            train_std, val_std, test_std,
            lat, lon, mean_lat, std_lat)


# ============================================================
# UČENJE MODELA ZA DANO KONFIGURACIJO
# ============================================================

def train_one_config(
    filters=(2, 4, 8),
    kernel_sizes=((5, 5), (3, 3), (3, 3)),
    activation="relu",
    use_periodic_padding=True,
    use_batchnorm=False,
    extra_conv_block=False,
    tag="baseline"
):
    """
    Natrenira CNN za izbrano konfiguracijo in:
      - shrani model
      - shrani zgodovino učenja
      - nariše learning curve
      - vrne model in pomožne podatke
    """
    (x_train, y_train,
     x_val,   y_val,
     x_test,  y_test,
     train_std, val_std, test_std,
     lat, lon, mean_lat, std_lat) = prepare_datasets()

    input_shape = x_train.shape[1:]  # (nlat, nlon, 1)

    model = create_cnn_model(
        input_shape=input_shape,
        filters=filters,
        kernel_sizes=kernel_sizes,
        activation=activation,
        use_periodic_padding=use_periodic_padding,
        use_batchnorm=use_batchnorm,
        extra_conv_block=extra_conv_block,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.3, patience=5, verbose=1
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[reduce_lr, early_stop],
        verbose=2,
    )

    # Shrani model in zgodovino
    model.save(f"model_{tag}.keras")
    np.save(f"history_{tag}.npy", history.history, allow_pickle=True)

    # Learning curve
    plt.figure()
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.yscale("log")
    plt.xlabel("Epoha")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"learning_{tag}.pdf"))
    plt.close()

    # Osnovni test na 1-dnevni napovedi
    test_loss, test_mae, test_mse = model.evaluate(x_test, y_test, verbose=0)
    print(f"[{tag}] Test MSE (1 dan naprej): {test_mse:.4e}")

    helper = {
        "train_std": train_std,
        "val_std":   val_std,
        "test_std":  test_std,
        "lat":       lat,
        "lon":       lon,
        "mean_lat":  mean_lat,
        "std_lat":   std_lat,
    }
    return model, history.history, helper


# ============================================================
# ITERATIVNA NAPOVED + METRIKE (RMSE, ACC)
# ============================================================

def destandardize(field_std, mean_lat, std_lat):
    """
    Pretvori polje iz standardiziranega v fizikalne enote:
        field = field_std * std_lat + mean_lat
    field_std shape: [..., nlat, nlon]
    mean_lat/std_lat: [1, nlat, 1]
    """
    return field_std * std_lat + mean_lat


def iterative_forecast(model, test_std, steps=7):
    """
    Izvede iterativno napoved na latitudinalno standardiziranih podatkih test_std:
      - test_std: [T, nlat, nlon] (standardizirano)
      - steps   : število dni naprej (npr. 7)

    Vrne:
      preds_std : [steps, Ncases, nlat, nlon]
      truth_std : [steps, Ncases, nlat, nlon]
    kjer je Ncases = T - steps.
    """
    T, nlat, nlon = test_std.shape
    Ncases = T - steps

    preds_std = np.zeros((steps, Ncases, nlat, nlon), dtype=np.float32)
    truth_std = np.zeros((steps, Ncases, nlat, nlon), dtype=np.float32)

    for t0 in range(Ncases):
        x_curr = test_std[t0][..., np.newaxis]  # (nlat, nlon, 1)

        for k in range(steps):
            # napovemo t0 + k + 1
            pred = model.predict(x_curr[np.newaxis, ...], verbose=0)  # (1,nlat,nlon,1)
            pred = pred[0, :, :, 0]  # (nlat, nlon)

            preds_std[k, t0] = pred
            truth_std[k, t0] = test_std[t0 + k + 1]

            # iterativno: napoved postane naslednji vhod
            x_curr = pred[..., np.newaxis]

    return preds_std, truth_std


def compute_rmse_acc(preds_phys, truth_phys):
    """
    Izračuna RMSE in ACC po lead time-u.

    preds_phys, truth_phys: [steps, Ncases, nlat, nlon] v Z500 enotah.

    RMSE: sqrt(mean( (pred-true)^2 ) čez vse primere in prostor)

    ACC (pattern correlation):
      - za vsak lead zložimo (Ncases, nlat, nlon) v (Ncases, nlat*nlon),
      - odštejemo prostorsko sredino (po vsaki točki posebej),
      - izračunamo korelacijo med pred in true po dimenziji (čas+prostor).
    """
    steps, Ncases, nlat, nlon = preds_phys.shape

    rmse = np.zeros(steps)
    acc  = np.zeros(steps)

    for k in range(steps):
        diff = preds_phys[k] - truth_phys[k]
        rmse[k] = np.sqrt(np.mean(diff**2))

        # ACC: pattern correlation
        pred_flat = preds_phys[k].reshape(Ncases, -1)
        true_flat = truth_phys[k].reshape(Ncases, -1)

        # odštejemo sredino po vsaki točki (anomalije)
        pred_anom = pred_flat - pred_flat.mean(axis=0, keepdims=True)
        true_anom = true_flat - true_flat.mean(axis=0, keepdims=True)

        num = np.sum(pred_anom * true_anom)
        den = np.sqrt(np.sum(pred_anom**2) * np.sum(true_anom**2))
        acc[k] = num / den if den != 0 else np.nan

    return rmse, acc


def evaluate_iterative(model, helper, tag="baseline", steps=7):
    """
    Izvede iterativno napoved za 'steps' dni naprej in izračuna
    RMSE/ACC (v fizikalnih enotah) po lead time-u.

    Rezultate nariše v figs/RMSE_ACC_{tag}.pdf
    """
    test_std = helper["test_std"]
    mean_lat = helper["mean_lat"]
    std_lat  = helper["std_lat"]

    preds_std, truth_std = iterative_forecast(model, test_std, steps=steps)

    # destandardiziraj na Z500 enote
    preds_phys = destandardize(preds_std, mean_lat, std_lat)
    truth_phys = destandardize(truth_std, mean_lat, std_lat)

    rmse, acc = compute_rmse_acc(preds_phys, truth_phys)

    leads = np.arange(1, steps + 1)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(leads, rmse, marker="o", label="RMSE")
    ax2.plot(leads, acc,  marker="s", label="ACC", linestyle="--")

    ax1.set_xlabel("Lead time (dni)")
    ax1.set_ylabel("RMSE (Z500 enote)")
    ax2.set_ylabel("ACC")

    ax1.grid(True, linestyle=":")
    ax1.set_xticks(leads)

    # Legenda
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"RMSE_ACC_{tag}.pdf"))
    plt.close()

    print(f"[{tag}] RMSE po lead time-u:", rmse)
    print(f"[{tag}] ACC  po lead time-u:", acc)


# ============================================================
# GLAVNA ZANKA: več konfiguracij za obvezni del naloge
# ============================================================

if __name__ == "__main__":

    # 1) Osnovna mreža + periodični padding + ReLU
    model_base, hist_base, helper_base = train_one_config(
        filters=(2, 4, 8),
        kernel_sizes=((5, 5), (3, 3), (3, 3)),
        activation="relu",
        use_periodic_padding=True,
        use_batchnorm=False,
        extra_conv_block=False,
        tag="base_periodic_relu"
    )
    evaluate_iterative(model_base, helper_base, tag="base_periodic_relu", steps=7)

    # 2) Več filtrov
    model_more_f, hist_more_f, helper_more_f = train_one_config(
        filters=(8, 16, 32),
        kernel_sizes=((5, 5), (3, 3), (3, 3)),
        activation="relu",
        use_periodic_padding=True,
        use_batchnorm=False,
        extra_conv_block=False,
        tag="more_filters"
    )
    evaluate_iterative(model_more_f, helper_more_f, tag="more_filters", steps=7)

    # 3) Večja jedra
    model_big_k, hist_big_k, helper_big_k = train_one_config(
        filters=(4, 8, 16),
        kernel_sizes=((7, 7), (5, 5), (3, 3)),
        activation="relu",
        use_periodic_padding=True,
        use_batchnorm=False,
        extra_conv_block=False,
        tag="bigger_kernels"
    )
    evaluate_iterative(model_big_k, helper_big_k, tag="bigger_kernels", steps=7)

    # 4) Druga aktivacija (ELU)
    model_elu, hist_elu, helper_elu = train_one_config(
        filters=(4, 8, 16),
        kernel_sizes=((5, 5), (3, 3), (3, 3)),
        activation="elu",
        use_periodic_padding=True,
        use_batchnorm=False,
        extra_conv_block=False,
        tag="elu_activation"
    )
    evaluate_iterative(model_elu, helper_elu, tag="elu_activation", steps=7)

    # 5) Zero padding namesto periodičnega
    model_zero, hist_zero, helper_zero = train_one_config(
        filters=(4, 8, 16),
        kernel_sizes=((5, 5), (3, 3), (3, 3)),
        activation="relu",
        use_periodic_padding=False,   # <<< IMPORTANT
        use_batchnorm=False,
        extra_conv_block=False,
        tag="zero_padding"
    )
    evaluate_iterative(model_zero, helper_zero, tag="zero_padding", steps=7)

    # 6) Dodatni conv+pool blok + BatchNorm
    model_extra, hist_extra, helper_extra = train_one_config(
        filters=(8, 16, 32),
        kernel_sizes=((5, 5), (3, 3), (3, 3)),
        activation="relu",
        use_periodic_padding=True,
        use_batchnorm=True,
        extra_conv_block=True,
        tag="extra_block_bn"
    )
    evaluate_iterative(model_extra, helper_extra, tag="extra_block_bn", steps=7)