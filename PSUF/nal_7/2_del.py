import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # <--- DODAJ TO VRSTICO

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
# pyrefly: ignore [missing-import]
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K

# 0. Ustvarimo mapo za grafe, če ta še ne obstaja
output_dir = "figs_2_del"
os.makedirs(output_dir, exist_ok=True)

# 1. Priprava podatkov MNIST
print("Nalagam MNIST podatke...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizacija na interval [0, 1] in preoblikovanje slik v 1D nize (784 pikslov)
original_dim = int(np.prod(x_train.shape[1:])) # 28 * 28 = 784
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape((len(x_train), original_dim))
x_test = x_test.reshape((len(x_test), original_dim))

print(f"Dimenzije učnih podatkov: {x_train.shape}")

# 2. Arhitektura VAE
hidden_dim = 64
latent_dim = 2

# --- ENKODER ---
inputs = keras.Input(shape=(original_dim,))
h = keras.layers.Dense(hidden_dim, activation='selu')(inputs)
z_mean = keras.layers.Dense(latent_dim)(h)
z_log_var = keras.layers.Dense(latent_dim)(h)

# Funkcija za vzorčenje (Reparametrizacijski trik)
def sampling(args):
    z_mean, z_log_var = args
    # Naključna spremenljivka iz normalne porazdelitve N(0, 1)
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.0, stddev=1.0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Sloj za vzorčenje
z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

# Združimo enkoder v model
encoder = keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')

# --- DEKODER ---
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = keras.layers.Dense(hidden_dim, activation='selu')(latent_inputs)
# Izhodna aktivacija je sigmoid, ker želimo piksle nazaj na intervalu [0,1]
outputs = keras.layers.Dense(original_dim, activation='sigmoid')(x)

decoder = keras.Model(latent_inputs, outputs, name='decoder')

# --- VAE ZDRUŽITEV IN CENOVNA FUNKCIJA ---
vae_outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, vae_outputs, name='vae')

# Cenovna funkcija = Rekonstrukcijska napaka + Kullback-Leibler divergenca
rec_loss = keras.losses.binary_crossentropy(inputs, vae_outputs)
rec_loss *= original_dim  # Množimo z 784 (vpliv obeh delov enačbe)
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

vae_loss = K.mean(rec_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# 3. Učenje modela (Naloga 2)
batch_size = 32
epochs = 100

print("Začenjam učenje VAE (to lahko traja nekaj minut, odvisno od strojne opreme)...")
# Odkomentiraj za učenje, zakomentiraj če želiš samo grafe
history = vae.fit(x_train, x_train, 
                  epochs=epochs, 
                  batch_size=batch_size, 
                  validation_data=None, # VAE na MNIST redko overfitta pri 2D latentnem prostoru
                  verbose=1)

# Shranimo uteži, da modela ni treba vedno znova učiti
vae.save_weights(os.path.join(output_dir, 'vae_weights.h5'))

# 4. VIZUALIZACIJE
print("Pripravljam grafe...")

# Naloga 3: Prikaz 2D latentnega prostora
# Preslikamo testne podatke v latentni prostor (uporabimo le srednje vrednosti z_mean)tukaj uporabimo x test
#z_test_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
# x_train:
z_test_mean, _, _ = encoder.predict(x_train, batch_size=batch_size)

plt.figure(figsize=(8, 7))
# Vsako številko pobarvamo drugače
#tukaj zamenjaj y_test in y_train 
scatter = plt.scatter(z_test_mean[:, 0], z_test_mean[:, 1], c=y_train, cmap='tab10', s=10, alpha=0.8)
# Dodamo legendo z diskretnimi barvami za vseh 10 števk
cbar = plt.colorbar(scatter, ticks=range(10))
cbar.set_label('Števka')
plt.xlabel(r"$z_{mean}[0]$")
plt.ylabel(r"$z_{mean}[1]$")
plt.title("2D Latentni prostor VAE modela")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, '1_latentni_prostor.pdf'), format='pdf', bbox_inches='tight')
plt.close()

# Naloga 4: Generiranje slik na 2D mreži
n = 15  # Velikost mreže (15 x 15 slik)
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# Mreža točk v latentnem prostoru, iz katere bomo generirali slike.
# Meje [-4, 4] so običajno dobre za N(0,1), ki ga VAE poskuša oponašati.
grid_x = np.linspace(-4, 4, n)
grid_y = np.linspace(4, -4, n) # Obrnemo y os, da se vizualno ujema s koordinatnim sistemom

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        # Dekoder pretvori koordinato nazaj v sliko
        x_decoded = decoder.predict(z_sample, verbose=0)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        
        # Sliko vstavimo na pravo mesto v veliki figuri
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
# Slike prikažemo črno-belo
plt.imshow(figure, cmap='gray')
plt.axis('off')
plt.title("Generirane števke iz 2D mreže latentnega prostora")
plt.savefig(os.path.join(output_dir, '2_generirana_mreza.pdf'), format='pdf', bbox_inches='tight')
plt.close()

print(f"Vsi grafi 2. dela so bili uspešno shranjeni v mapo '{output_dir}/' v formatu PDF!")