#%%
"""
Avtorja: Boštjan Melinc in Uroš Perkan.
Načeloma je dovolj, da spreminjate samo vrstice, ki so med vrsticama
# ... zacni urejanje
in
# ... koncaj urejanje
"""
import pickle

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, Conv2D, BatchNormalization
import cartopy.crs as ccrs
import os
import gc

#%%
# ====================================================
#            IZBOR KOLICIN IN RESOLUCIJE
# ====================================================

# ... zacni urejanje

# Kolicine na voljo: ('Z500', 'T2m', 'MSLP')
# Oznake:
# Z500: geopotencial na 500 hPa ploskvi,
# T2m: temperatura na 2m,
# MSLP: tlak pri tleh, preracunan na nivo morja
kolicine = ['Z500']

# Casovni korak (v dnevih)
casovni_korak = 1

# Resolucije na voljo: 2, 4, 10
# Enota: stopinja geografske dolzine in sirine
resolucija = 4

# Ali upostevam masko kopnega in morja?
kopno_morje = False

# Podatki so na voljo za leta 1940-2023
training_set_leta_min_max = (1979, 2017)
validation_set_leta_min_max = (2018, 2022)
test_set_leta_min_max = (2023, 2023)


# Standardizacija podatkov
# Na voljo:
# 'latitudinalna': razlicna standardizacija za vsako geografsko sirino
# 'globalna': ista standardizacija za celo Zemljo
standardizacija_podatkov = 'latitudinalna'

# Lokacija podatkov
mapa_s_podatki = '/home/jurijs/Documents/Faks/PSUF/nal_4/DEL_2/data'

# ... koncaj urejanje

# ====================================================
#           PRIPRAVA PODATKOV ZA TRENING
# ====================================================


def standardizacija(podatki, kolicine=kolicine, resolucija=resolucija, standardizacija_podatkov=standardizacija_podatkov):
    if standardizacija_podatkov == 'globalna':
        klimatolosko_povprecje = []
        klimatoloski_std = []
        for kolicina in kolicine:
            povp_in_std = pickle.load(open(
                mapa_s_podatki + f'/climatological_mean_and_std/{kolicina}_{resolucija}_global_climatological_mean_and_std.p', 'rb'
            ))
            klimatolosko_povprecje.append(povp_in_std[0])
            klimatoloski_std.append(povp_in_std[1])
        return (podatki - np.array(klimatolosko_povprecje)) / np.array(klimatoloski_std)
    elif standardizacija_podatkov == 'latitudinalna':
        klimatoloska_povprecja = []
        klimatoloski_stdji = []
        for kolicina in kolicine:
            povp_in_std = pickle.load(open(
                mapa_s_podatki + f'/climatological_mean_and_std/{kolicina}_{resolucija}_latitudinal_climatological_mean_and_std.p',
                'rb'
            ))
            lat, lon = np.shape(povp_in_std[0])
            klimatoloska_povprecja.append(np.expand_dims(povp_in_std[0], (0, 3)))
            klimatoloski_stdji.append(np.expand_dims(povp_in_std[1], (0, 3)))
        klimatolosko_povprecje = np.concatenate(np.array(klimatoloska_povprecja), axis=-1)
        klimatoloski_std = np.concatenate(np.array(klimatoloski_stdji), axis=-1)
        return (podatki - klimatolosko_povprecje) / klimatoloski_std
    else:
        raise AttributeError # Neznana standardizacija podatkov

def destandardizacija(standardizirani_podatki, kolicine=kolicine, resolucija=resolucija, standardizacija_podatkov=standardizacija_podatkov, kopno_morje=kopno_morje):
    if kopno_morje:
        standardizirani_podatki = standardizirani_podatki[...,:-1]
    if standardizacija_podatkov == 'globalna':
        klimatolosko_povprecje = []
        klimatoloski_std = []
        for kolicina in kolicine:
            povp_in_std = pickle.load(open(
                mapa_s_podatki + f'/climatological_mean_and_std/{kolicina}_{resolucija}_global_climatological_mean_and_std.p', 'rb'
            ))
            klimatolosko_povprecje.append(povp_in_std[0])
            klimatoloski_std.append(povp_in_std[1])
        return standardizirani_podatki * np.array(klimatoloski_std) + np.array(klimatolosko_povprecje)
    elif standardizacija_podatkov == 'latitudinalna':
        klimatoloska_povprecja = []
        klimatoloski_stdji = []
        for kolicina in kolicine:
            povp_in_std = pickle.load(open(
                mapa_s_podatki + f'/climatological_mean_and_std/{kolicina}_{resolucija}_latitudinal_climatological_mean_and_std.p',
                'rb'
            ))
            klimatoloska_povprecja.append(np.expand_dims(povp_in_std[0], (0, 3)))
            klimatoloski_stdji.append(np.expand_dims(povp_in_std[1], (0, 3)))
        klimatolosko_povprecje = np.concatenate(np.array(klimatoloska_povprecja), axis=-1)
        klimatoloski_std = np.concatenate(np.array(klimatoloski_stdji), axis=-1)
        return standardizirani_podatki * np.array(klimatoloski_std) + np.array(klimatolosko_povprecje)
    else:
        raise AttributeError # Neznana standardizacija podatkov



def dodeli_standardizirane_podatke(leta_min_max, kolicine=kolicine, resolucija=resolucija, standardizacija_podatkov=standardizacija_podatkov):
    nc_koda = {'Z500':'z', 'T2m':'2t', 'MSLP':'msl'}
    izhodni_podatki = []
    for leto in range(leta_min_max[0], leta_min_max[1] + 1):
        podatki_za_to_kolicino = []
        for kolicina in kolicine:
            nc_datoteka = netCDF4.Dataset(mapa_s_podatki + f'/{kolicina}/{kolicina}_{resolucija}_{leto}.nc')
            if kolicina == 'Z500':
                d = nc_datoteka[nc_koda[kolicina]][:, 0]
            else:
                d = nc_datoteka[nc_koda[kolicina]][:]
            podatki_za_to_kolicino.append(np.expand_dims(d, (-1)))
        standardizirani_podatki = standardizacija(np.concatenate([p for p in podatki_za_to_kolicino], axis=-1), standardizacija_podatkov=standardizacija_podatkov)

        if kopno_morje:
            lsm = np.expand_dims(pickle.load(open(mapa_s_podatki + f'/LSM/LSM_{resolucija}.p', 'rb')), axis=(0, -1))
            lsm = np.concatenate([lsm for i in range(len(standardizirani_podatki))], axis=0)
            standardizirani_podatki = np.concatenate((standardizirani_podatki, lsm), axis=-1)

        if leta_min_max[1] > leta_min_max[0]:
            izhodni_podatki.append(standardizirani_podatki)
        else:
            return standardizirani_podatki

    izhodni_podatki = np.concatenate(izhodni_podatki, axis=0)

    return izhodni_podatki




# PODATKI ZA TRENING IN VALIDACIJO
print('DODELJEVANJE PODATKOV TRAIN')
podatki_train = dodeli_standardizirane_podatke(training_set_leta_min_max)
x_train, y_train = podatki_train[:-casovni_korak], podatki_train[casovni_korak:]

del podatki_train
gc.collect()

if kopno_morje:
    y_train = y_train[..., :-1]  # Polja kopnega in morja ne napovedujemo!

print('DODELJEVANJE PODATKOV VALIDATION')
podatki_validation = dodeli_standardizirane_podatke(validation_set_leta_min_max)
x_validation, y_validation = podatki_validation[:-casovni_korak], podatki_validation[casovni_korak:]
if kopno_morje:
    y_validation = y_validation[..., :-1]  # Polja kopnega in morja ne napovedujemo!
del podatki_validation
gc.collect()


# ====================================================
#                   PRIPRAVA MODELA
# ====================================================
from keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
class PeriodicPadding(layers.Layer):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def call(self, input):
        NorthSouth_pad = self.kernel_size[0]//2
        EastWest_pad = self.kernel_size[1]//2

        # North/South
        if NorthSouth_pad > 0:
            top = tf.reverse(input[:,0:NorthSouth_pad,:,:], axis=(-3,))
            top = tf.roll(top, shift=int(top.shape[-2]/2), axis=-2)

            bottom = tf.reverse(input[:,-NorthSouth_pad:,:,:], axis=(-3,))
            bottom = tf.roll(bottom, shift=int(bottom.shape[-2]/2), axis=-2)

            arr = tf.concat((top, input, bottom), axis=-3)
        else:
            arr = input

        # East/West
        if EastWest_pad > 0:
            left = arr[:,:,0:EastWest_pad,:]
            right = arr[:,:,-EastWest_pad:,:]
            arr = tf.concat((right, arr, left), axis=-2)

        return arr


# ... zacni urejanje

filters= 8,16,32 # ali 16,32,64 ali 8,16,32 ali 2,4,8

K = 3 # 3 ali 5 ali 9

act = 'elu' # ali 'relu', 'tanh', 'elu', 'sigmoid'


def create_model_CNN(filters = filters, K = K, act = act):
    # Dimenzija vhoda naj bo (batch_size, nlat, nlon, st. polj)
    # Pri resoluciji 4 stopinje je nlat=45, nlon=90
    # Tu se k stevilu polj steje tudi maska kopnega in morja!
    # Za preverjanje teh 3 stevil si naprintajte npr. np.shape(x_train)
    input_state = keras.Input(shape=(45,90,1)) # <-- doloci resolucijo in število vhodnih kanalov
    # ENCODER
    x = Conv2D(filters=filters[0], kernel_size=(K, K), activation=act, 
               kernel_initializer='glorot_uniform', padding="valid")(PeriodicPadding((5,5))(input_state))
    
    # za kernel ko je 9

    x = Conv2D(filters=8, kernel_size=(9,9),
           activation=act, padding="same")(input_state)
    
    # x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3))(x)
    x = Conv2D(filters=filters[1], kernel_size=(3, 3), padding="same", 
               kernel_initializer='glorot_uniform', activation=act,)(x)
    x = MaxPooling2D((3, 3))(x)
    encoded = Conv2D(filters=filters[2], kernel_size=(3, 3), padding="same", 
                     kernel_initializer='glorot_uniform', activation=act,)(x)

    # DECODER
    x = UpSampling2D((3, 3))(encoded)
    x = layers.Conv2D(filters=filters[1], kernel_size=(3, 3), padding="same", 
                      kernel_initializer='glorot_uniform', activation=act)(x)
    x = UpSampling2D((3, 3))(x)
    # V zadnji plasti naj bo toliko filtrov, kot je izhodnih polj (polja kopnega in morja NE napovedujemo!)
    decoded = layers.Conv2D(filters=1, kernel_size=(3, 3), padding="same",  # <-- doloci število izhodnih kanalov
                            kernel_initializer='glorot_uniform', activation='linear')(x)


    encoder_decoder = keras.Model(input_state, decoded)
    print(encoder_decoder.summary())
    encoder_decoder.compile(
            optimizer=keras.optimizers.Adam(),
            loss="mean_squared_error", 
            metrics=['mae', 'mse'])

    return encoder_decoder

def create_model_CNN_extra():
    input_state = keras.Input(shape=(45,90,1))

    # ENCODER
    x = Conv2D(8,(5,5),activation='relu',padding="valid")(PeriodicPadding((5,5))(input_state))
    x = Conv2D(8,(3,3),activation='relu',padding="same")(x)   # dodatna plast
    x = MaxPooling2D((3,3))(x)

    x = Conv2D(16,(3,3),activation='relu',padding="same")(x)
    x = Conv2D(16,(3,3),activation='relu',padding="same")(x)  # dodatna plast
    x = MaxPooling2D((3,3))(x)

    encoded = Conv2D(32,(3,3),activation='relu',padding="same")(x)

    # DECODER — simetrično
    x = UpSampling2D((3,3))(encoded)
    x = Conv2D(16,(3,3),activation='relu',padding="same")(x)
    x = Conv2D(16,(3,3),activation='relu',padding="same")(x)

    x = UpSampling2D((3,3))(x)
    x = Conv2D(8,(3,3),activation='relu',padding="same")(x)
    decoded = Conv2D(1,(3,3),activation='relu',padding="same")(x)

    model = keras.Model(input_state, decoded)
    model.compile(optimizer=keras.optimizers.Adam(), loss="mse", metrics=['mae','mse'])
    return model

def create_model_CNN_padding():
    input_state = keras.Input(shape=(45,90,1))

    # ENCODER
    x = Conv2D(8,(5,5),activation='relu',padding="same")(input_state)
    x = Conv2D(8,(3,3),activation='relu',padding="same")(x)   # dodatna plast
    x = MaxPooling2D((3,3))(x)

    x = Conv2D(16,(3,3),activation='relu',padding="same")(x)
    x = Conv2D(16,(3,3),activation='relu',padding="same")(x)  # dodatna plast
    x = MaxPooling2D((3,3))(x)

    encoded = Conv2D(32,(3,3),activation='relu',padding="same")(x)

    # DECODER — simetrično
    x = UpSampling2D((3,3))(encoded)
    x = Conv2D(16,(3,3),activation='relu',padding="same")(x)
    x = Conv2D(16,(3,3),activation='relu',padding="same")(x)

    x = UpSampling2D((3,3))(x)
    x = Conv2D(8,(3,3),activation='relu',padding="same")(x)
    decoded = Conv2D(1,(3,3),activation='relu',padding="same")(x)

    model = keras.Model(input_state, decoded)
    model.compile(optimizer=keras.optimizers.Adam(), loss="mse", metrics=['mae','mse'])
    return model
# ... koncaj urejanje


# =========================================
#       CALLBACKS
# =========================================

# EARLY STOP
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=8, verbose=0, mode='min')  # Manjsi patience, prej se ustavi

# REDUCE LEARNING RATE ON PLATEAU
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.4, patience=10, min_lr=1e-5)

# ... zacni urejanje
batch_size = 32
epochs = 30
# ... koncaj urejanje

# %%
# =========================================
#       FITANJE
# =========================================

# CNN example
# ... zacni urejanje
path_model_CNN =  '/home/jurijs/Documents/Faks/PSUF/nal_4/DEL_2/models/cnn_model.keras' # !NASTAVI SVOJO POT SHRANJEVANJA MODELA! path_model_CNN =
path_history_CNN = '/home/jurijs/Documents/Faks/PSUF/nal_4/DEL_2/history/history' # !NASTAVI SVOJO POT SHRANJEVANJA ZGODOVINE TRENIRANJA!
# ... koncaj urejanje

# ... zacni urejanje
model_CNN = create_model_CNN()
History_CNN = model_CNN.fit(x=x_train, y=y_train, 
                            validation_data=(x_validation, y_validation),
                            batch_size=batch_size, 
                            epochs=epochs,
                            callbacks=[reduce_lr, early_stop],
                            shuffle=True)  # Treniranje modela

loss_CNN = History_CNN.history['loss']
val_loss_CNN = History_CNN.history['val_loss']

model_CNN.save(path_model_CNN)
np.savez(path_history_CNN, loss=loss_CNN, val_loss=val_loss_CNN)
# ... koncaj urejanje



# Brisanje trening podatkov, ki jih ne potrebujemo vec
del x_train, y_train, x_validation, y_validation
gc.collect()

# =========================================
#       NALAGANJE NATRENIRANIH MODELOV
# =========================================
# ... zacni urejanje
model_CNN = tf.keras.models.load_model(path_model_CNN)
History_CNN = np.load(path_history_CNN + '.npz')
# print(model_CNN.summary())
# ... koncaj urejanje

##########################################
# %%=====================================
#    GRAFICNI PRIKAZ UCENJA NN
# =======================================

# ... zacni urejanje
Epochs_CNN = [i for i in range(1, len(History_CNN['loss']) + 1)]

fig = plt.figure()

gs = fig.add_gridspec(1, 1)

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title(f'Potek ucenja - extra)') # Kernel ({K}x{K}) Filtri {filters}
ax1.set_xlabel('epoch')
ax1.set_ylabel('val_loss = val_mean_squared_error')
ax1.plot(Epochs_CNN, History_CNN['val_loss'], label=r'CNN')
ax1.legend(loc='upper right')

plt.tight_layout()
#plt.savefig(f"figs/CNN_and_ResNN_learning_rate_NN_extra.pdf") # _kernel{K} _filters({filters})
# ... koncaj urejanje
#plt.show()
plt.close()



# %%=====================================
# =======================================
#              NAPOVED
# =======================================
# =======================================

# Nalaganje testnih podatkov
podatki_test = dodeli_standardizirane_podatke(test_set_leta_min_max)
x_test = podatki_test[:]

del podatki_test
x_test_dest = destandardizacija(x_test)
gc.collect()

# =======================================
#     VREMENSKE NAPOVEDI
# =======================================

# N = stevilo napovedi
# forecast_time_steps = napoved za koliko casovnih korakov

# ... zacni urejanje
forecast_time_steps = 7 # število dni
Nmax = np.shape(x_test)[0] - forecast_time_steps * casovni_korak   # najvecji mozen N, za katerega se imamo dovolj podatkov
N = Nmax
# ... koncaj urejanje


# =======================================
# NAPOVED KLIMATOLOGIJE IN PERSISTENCE
# =======================================

X_resnica_dest = np.array([[x_test_dest[zac_dan + forecast_time_step*casovni_korak] for zac_dan in range(N)]
                        for forecast_time_step in range(forecast_time_steps + 1)])

persistenca = np.array([[x_test_dest[zac_dan] for zac_dan in range(N)]
                        for forecast_time_step in range(forecast_time_steps + 1)])
klimatologija = []
for kolicina in kolicine:
    povp_in_std = pickle.load(open(
        mapa_s_podatki + f'/climatological_mean_and_std/{kolicina}_{resolucija}_daily_latlon_climatological_mean_and_std.p',
        'rb'
    ))
    povp = np.expand_dims(np.array(povp_in_std)[0], axis=-1)
    klimatologija.append(povp)
klimatologija = np.concatenate(klimatologija, axis=-1)
klimatoloska_napoved = np.array([[klimatologija[zac_dan + forecast_time_step] for zac_dan in range(N)]
                                    for forecast_time_step in range(forecast_time_steps + 1)])

# =======================================
#      NAPOVED NEVRONSKE MREZE
# =======================================

def napoved_NN(model, input_states, forecast_time_steps):
    if kopno_morje: # Treba je sproti dodajat polje kopnega in morja
        lsm = np.expand_dims(pickle.load(open(mapa_s_podatki + f'/LSM/LSM_{resolucija}.p', 'rb')), axis=(0, -1))
        lsm = np.concatenate([lsm for i in range(len(input_states))], axis=0)
    stara_stanja = []
    stara_destandardizirana_stanja = []
    stara_stanja.append(input_states)
    stara_destandardizirana_stanja.append(destandardizacija(input_states))
    for casovni_korak in range(1, forecast_time_steps + 1):
        nova_stanja = model.predict(stara_stanja[-1])
        if kopno_morje: # Treba je sproti dodajat polje kopnega in morja
            nova_stanja = np.concatenate((nova_stanja, lsm), axis=-1)
        stara_stanja.append(nova_stanja)
        stara_destandardizirana_stanja.append(destandardizacija(nova_stanja))

    return np.array(stara_destandardizirana_stanja)

# # ... zacni urejanje
napoved = napoved_NN(model=model_CNN, input_states=x_test[:N], forecast_time_steps=forecast_time_steps)
# ... koncaj urejanje


# =======================================
#      METRIKE USPESNOSTI
# =======================================

# Geografske dolzine
lons = np.arange(-180 + resolucija/2, 180 - resolucija/2 + 1e-5, step=resolucija)
# Geografske sirine
lats = np.arange(90 - resolucija/2, -90 + resolucija/2 - 1e-5, step=-resolucija)

def RMSE(napoved, resnica):
    rmse = np.sqrt(np.mean((napoved - resnica)**2))

    return rmse

def ACC(napoved, resnica, klimatolosko_povprecje):
    stevec = np.mean((napoved - klimatolosko_povprecje) * (resnica - klimatolosko_povprecje))
    imenovalec = np.sqrt(np.mean((napoved - klimatolosko_povprecje)**2) * np.mean((resnica - klimatolosko_povprecje)**2))

    return stevec / imenovalec

def metrike_uspesnosti(napoved, resnica, klimatologija, compute_ACC=True):
    """Izracuna RMSE in ACC za izbrano napoved.
    Ce racunas metrike za klimatolosko napoved, nastavi compute_ACC=False (v nasprotnem primeru se zgodi ZeroDivisionError)"""
    RMSE_all = []
    ACC_all = []
    for ikolicina in range(len(kolicine)):
        RMSE1 = []
        ACC1 = []

        for cas_napovedi in range(len(napoved)):
            RMSE1.append(RMSE(napoved[cas_napovedi][...,ikolicina], resnica[cas_napovedi][...,ikolicina]))
            if compute_ACC:
                ACC1.append(ACC(napoved[cas_napovedi][...,ikolicina], resnica[cas_napovedi][...,ikolicina], klimatologija[cas_napovedi][...,ikolicina]))

        RMSE_all.append(RMSE1)
        ACC_all.append(ACC1)

    return np.array(RMSE_all), np.array(ACC_all)
    



# # ... zacni urejanje
RMSE_napoved, ACC_napoved = metrike_uspesnosti(napoved=napoved, resnica=X_resnica_dest, klimatologija=klimatoloska_napoved)
print('RMSE_napoved', RMSE_napoved)
# ... koncaj urejanje

RMSE_klimatologija, _ = metrike_uspesnosti(napoved=klimatoloska_napoved, resnica=X_resnica_dest, klimatologija=klimatoloska_napoved, compute_ACC=False)
RMSE_persistenca, ACC_persiscenca = metrike_uspesnosti(napoved=persistenca, resnica=X_resnica_dest, klimatologija=klimatoloska_napoved)


print('RMSE_klimatologija', RMSE_klimatologija)
print('RMSE_persistenca', RMSE_persistenca)


# =====================================
#       RISANJE METRIK USPESNOSTI
# =====================================

for ikolicina in range(len(kolicine)):
    t_predict = [i*casovni_korak for i in range(forecast_time_steps + 1)]

    fig = plt.figure(figsize=(6, 8))

    gs = fig.add_gridspec(2, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlabel(r'Dan napovedi')
    ax1.set_ylabel(r'RMSE')
    ax1.set_title(f'Uspešnost napovedi {kolicine[ikolicina]} - extra) ') # Kernel ({K}x{K}) Filtri ({filters})
    ax1.grid()


    # ... zacni urejanje
    ax1.plot(t_predict, RMSE_napoved[ikolicina], linewidth=2, linestyle='solid', \
             label=r'napoved NN')
    # ... koncaj urejanje

    ax1.plot(t_predict, RMSE_persistenca[ikolicina], color='black', linewidth=2, linestyle='solid', \
             label=r'persistenca')
    ax1.plot(t_predict, RMSE_klimatologija[ikolicina], color='black', linewidth=2, linestyle='dashed', \
             label=r'klimatologija')

    ax1.set_xticks(ticks=t_predict)


    ax2 = fig.add_subplot(gs[1, 0])
    ax2.grid()
    ax2.set_ylabel(r'ACC')
    ax2.set_xlabel(r'Dan napovedi')
    ax2.set_xticks(ticks=t_predict)

    # ... zacni urejanje
    ax2.plot(t_predict, ACC_napoved[ikolicina], linewidth=2, linestyle='solid', \
             label=r'CNN napoved')
    # ... koncaj urejanje

    ax2.plot(t_predict, ACC_persiscenca[ikolicina], color='black', linewidth=2, linestyle='-', \
             label=r'persistenca')
    ax2.plot(t_predict, [0 for i in t_predict], color='black', linewidth=2, linestyle='--', \
             label=r'klimatologija')
    ax2.axhline(0.6, linestyle=':', color='grey')


    ax2.legend(loc='upper right')

    plt.tight_layout()

    fig.patch.set_facecolor('white')

    # ... zacni urejanje
    #plt.savefig(f"figs/CNN_RMSE_ACC_{kolicine[ikolicina]}_extra.pdf") #_kernel{K}  _filters{filters}
    # ... koncaj urejanje
    #plt.show()
    plt.close()


# =====================================
#       RISANJE NAPOVEDANIH POLJ
# =====================================

def plot_meteorological_field(meteorological_field, kolicina, naslov):
    vmins = {'Z500':42000, 'T2m':190, 'MSLP':92000}
    vmaxs = {'Z500':60000, 'T2m':320, 'MSLP':108000}
    cmaps = {'Z500':'nipy_spectral', 'T2m':'terrain', 'MSLP':'bwr'}

    lt = lats
    ln = lons
    lns, lts = np.meshgrid(ln, lt)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    # V primeru napake "AttributeError: 'GeoAxesSubplot' object has no attribute '_autoscaleXon'"
    # odkomentiraj naslednji dve vrstici (napaka se lahko zgodi zaradi spora med verzijama matplotlib in cartopy)
    #ax._autoscaleXon = False
    #ax._autoscaleYon = False

    filled_c = ax.contourf(lns, lts, meteorological_field, np.linspace(vmins[kolicina], vmaxs[kolicina], 30), cmap=cmaps[kolicina],
                          transform=ccrs.PlateCarree())# np.linspace(vmins[kolicina], vmaxs[kolicina], 30),
    line_c = ax.contour(lns, lts, meteorological_field, levels=filled_c.levels, linewidths=0.4, colors=['black'],
                        transform=ccrs.PlateCarree())


    ax.coastlines()
    ax.gridlines()
    ax.set_global()

    ax.set_title(naslov)
    fig.colorbar(filled_c, ax=ax, fraction=0.045)
    fig.tight_layout()
    #plt.savefig('figs/fig.png')
    plt.show()

# ... zacni urejanje
for ikolicina in range(len(kolicine)):
    for i in range(N):
        print(i, N)
        for j in range(forecast_time_steps):
            plot_meteorological_field(napoved[j][i][..., ikolicina], kolicina=kolicine[ikolicina], naslov=f"{kolicine[ikolicina]}, {j} dni od zacetnega dne {i}")
        break   # Ta break zagotovi, da risemo samo en cikel napovedi
# ... koncaj urejanje

