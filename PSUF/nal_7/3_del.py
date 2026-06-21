import os
# Obvezno za Keras 3 kompatibilnost s staro kodo
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

# ==========================================
# 0. PRIPRAVA DIREKTORIJA
# ==========================================
output_dir = "figs_3_del"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 1. NALAGANJE PODATKOV
# ==========================================
print("Nalagam podatke...")
data_dir = "Podatki_7naloga"

# Testni podatki (H - 10% signala)
x_H_orig = np.load(os.path.join(data_dir, "lhco_H_jetobs.npy")).astype('float32')
invmass_H = np.load(os.path.join(data_dir, "lhco_H_invmass.npy"))
labels_H = np.load(os.path.join(data_dir, "lhco_H_labels.npy"))

# Podatki črne škatle
x_bb_orig = np.load(os.path.join(data_dir, "blackbox_jetobs.npy")).astype('float32')
invmass_bb = np.load(os.path.join(data_dir, "blackbox_invmass.npy"))

# ==========================================
# 2. PREDPROCESIRANJE (Naloga 5)
# ==========================================
print("Izvajam QuantileTransformer transformacijo...")
scaler_H = QuantileTransformer(output_distribution='uniform')
x_H_trans = scaler_H.fit_transform(x_H_orig)

# Graf 1: Primerjava pred in po transformaciji za prve 4 spremenljivke
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
imena_spremenljivk = [r'$m_{j1}$', r'$(\tau_2/\tau_1)_1$', r'$(\tau_3/\tau_2)_1$', r'$m_{d_{j1}}$']

for i in range(4):
    # Pred transformacijo
    axes[0, i].hist(x_H_orig[:, i], bins=50, color='blue', alpha=0.7)
    axes[0, i].set_title(f'Original: {imena_spremenljivk[i]}')
    # Po transformaciji
    axes[1, i].hist(x_H_trans[:, i], bins=50, color='orange', alpha=0.7)
    axes[1, i].set_title(f'Transformirano: {imena_spremenljivk[i]}')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '1_transformacija_Naloga5.pdf'))
plt.close()

# ==========================================
# 3. FUNKCIJA ZA GRADNJO VAE
# ==========================================
def build_vae(original_dim=8, hidden_dim=64, latent_dim=1):
    inputs = keras.Input(shape=(original_dim,))
    h = keras.layers.Dense(hidden_dim, activation='selu')(inputs)
    h = keras.layers.Dense(hidden_dim, activation='selu')(h)
    h = keras.layers.Dense(hidden_dim, activation='selu')(h)
    z_mean = keras.layers.Dense(latent_dim)(h)
    z_log_var = keras.layers.Dense(latent_dim)(h)

    def sampling(args):
        z_m, z_lv = args
        epsilon = K.random_normal(shape=(K.shape(z_m)[0], latent_dim), mean=0.0, stddev=1.0)
        return z_m + K.exp(0.5 * z_lv) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')

    latent_inputs = keras.Input(shape=(latent_dim,))
    h_dec = keras.layers.Dense(hidden_dim, activation='selu')(latent_inputs)
    h_dec = keras.layers.Dense(hidden_dim, activation='selu')(h_dec)
    h_dec = keras.layers.Dense(hidden_dim, activation='selu')(h_dec)
    outputs = keras.layers.Dense(original_dim, activation='linear')(h_dec) # Linearna aktivacija!

    decoder = keras.Model(latent_inputs, outputs, name='decoder')
    vae_outputs = decoder(encoder(inputs)[2])
    vae = keras.Model(inputs, vae_outputs, name='vae')

    # Cenovna funkcija
    rec_loss = keras.losses.mean_squared_error(inputs, vae_outputs)
    rec_loss *= 5000.0
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(rec_loss + kl_loss)
    vae.add_loss(vae_loss)

    vae.compile(optimizer='adadelta')
    return vae, encoder

# ==========================================
# 4. UČENJE NA TESTNIH PODATKIH (Naloga 6)
# ==========================================
print("\n--- Gradnja in učenje VAE na TESTNIH podatkih (H) ---")
vae_H, encoder_H = build_vae()
vae_H.fit(x_H_trans, x_H_trans, epochs=100, batch_size=1000, verbose=1)

# ==========================================
# 5. ANALIZA LATENTNEGA PROSTORA IN ROC (Naloge 7, 8)
# ==========================================
print("Pripravljam analizo latentnega prostora...")
z_mean_H, z_logvar_H, _ = encoder_H.predict(x_H_trans, batch_size=1000)

z_mean_H = z_mean_H.flatten()
z_logvar_H = z_logvar_H.flatten()
z_mean_sq_H = np.square(z_mean_H)

# Graf 2: Porazdelitev v latentnem prostoru (Naloga 7)
mask_sig = (labels_H == 1)
mask_bkg = (labels_H == 0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(z_mean_H[mask_bkg], bins=50, alpha=0.5, label='Ozadje', density=True)
axes[0].hist(z_mean_H[mask_sig], bins=50, alpha=0.8, label='Signal', density=True)
axes[0].set_title(r'$z_{mean}$')
axes[0].legend()

axes[1].hist(z_logvar_H[mask_bkg], bins=50, alpha=0.5, label='Ozadje', density=True)
axes[1].hist(z_logvar_H[mask_sig], bins=50, alpha=0.8, label='Signal', density=True)
axes[1].set_title(r'$z_{logvar}$')
axes[1].legend()

axes[2].hist(z_mean_sq_H[mask_bkg], bins=50, alpha=0.5, label='Ozadje', density=True)
axes[2].hist(z_mean_sq_H[mask_sig], bins=50, alpha=0.8, label='Signal', density=True)
axes[2].set_title(r'$z_{mean}^2$')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_latentni_prostor_Naloga7.pdf'))
plt.close()

# ==========================================
# Graf 3: ROC krivulje (Naloga 8)
# ==========================================
# 1. Za z_mean^2 avtomatsko izberemo predznak, ki da NAJVEČJI AUC
auc_zm_pos = metrics.roc_auc_score(labels_H, z_mean_sq_H)
auc_zm_neg = metrics.roc_auc_score(labels_H, -z_mean_sq_H)

if auc_zm_pos >= auc_zm_neg:
    auc_zm = auc_zm_pos
    fpr_zm, tpr_zm, _ = metrics.roc_curve(labels_H, z_mean_sq_H)
    label_zm = r'$+z_{mean}^2$'
else:
    auc_zm = auc_zm_neg
    fpr_zm, tpr_zm, _ = metrics.roc_curve(labels_H, -z_mean_sq_H)
    label_zm = r'$-z_{mean}^2$'

# 2. Za +z_logvar
auc_zl_pos = metrics.roc_auc_score(labels_H, z_logvar_H)
fpr_zl_pos, tpr_zl_pos, _ = metrics.roc_curve(labels_H, z_logvar_H)
label_zl_pos = r'$+z_{logvar}$'

# 3. Za -z_logvar
auc_zl_neg = metrics.roc_auc_score(labels_H, -z_logvar_H)
fpr_zl_neg, tpr_zl_neg, _ = metrics.roc_curve(labels_H, -z_logvar_H)
label_zl_neg = r'$-z_{logvar}$'

# --- RISANJE GRAFOV ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Levi graf: ROC Krivulja
axes[0].plot(fpr_zm, tpr_zm, label=f'{label_zm} (AUC = {auc_zm:.3f})', color='red')
axes[0].plot(fpr_zl_pos, tpr_zl_pos, label=f'{label_zl_pos} (AUC = {auc_zl_pos:.3f})', color='blue')
axes[0].plot(fpr_zl_neg, tpr_zl_neg, label=f'{label_zl_neg} (AUC = {auc_zl_neg:.3f})', color='green')
axes[0].plot([0, 1], [0, 1], 'k--')
axes[0].set_xlabel('FPR')
axes[0].set_ylabel('TPR')
axes[0].set_title('ROC Krivulja')
axes[0].legend()

# Desni graf: 1/FPR v odvisnosti od TPR
valid_zm = fpr_zm > 0
valid_zl_pos = fpr_zl_pos > 0
valid_zl_neg = fpr_zl_neg > 0

axes[1].plot(tpr_zm[valid_zm], 1.0 / fpr_zm[valid_zm], color='red', label=label_zm)
axes[1].plot(tpr_zl_pos[valid_zl_pos], 1.0 / fpr_zl_pos[valid_zl_pos], color='blue', label=label_zl_pos)
axes[1].plot(tpr_zl_neg[valid_zl_neg], 1.0 / fpr_zl_neg[valid_zl_neg], color='green', label=label_zl_neg)
axes[1].set_yscale('log')
axes[1].set_xlabel('TPR')
axes[1].set_ylabel('1 / FPR')
axes[1].set_title('1/FPR v odvisnosti od TPR')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_roc_krivulje_Naloga8.pdf'))
plt.close()

# ==========================================
# 6. DOLOČANJE MAS (Naloga 9)
# ==========================================
def narisi_mase(klasifikator, original_x, original_invmass, naslov_grafa, ime_datoteke, invmass_range=(2500, 4500)):
    sorted_idx = np.argsort(klasifikator)[::-1]

    idx_500 = sorted_idx[:500]
    idx_1000 = sorted_idx[:1000]
    idx_2000 = sorted_idx[:2000]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # Masa 1. curka
    axs[0].hist(original_x[idx_2000, 0], bins=40, alpha=0.3, color='blue', label='Bottom 2000')
    axs[0].hist(original_x[idx_1000, 0], bins=40, alpha=0.5, color='green', label='Bottom 1000')
    axs[0].hist(original_x[idx_500, 0], bins=40, alpha=0.8, color='red', label='Bottom 500')
    axs[0].set_title(r'Masa 1. curka ($m_{j1}$)')
    axs[0].legend()

    # Masa 2. curka
    axs[1].hist(original_x[idx_2000, 4], bins=40, alpha=0.3, color='blue', label='Bottom 2000')
    axs[1].hist(original_x[idx_1000, 4], bins=40, alpha=0.5, color='green', label='Bottom 1000')
    axs[1].hist(original_x[idx_500, 4], bins=40, alpha=0.8, color='red', label='Bottom 500')
    axs[1].set_title(r'Masa 2. curka ($m_{j2}$)')
    axs[1].legend()

    # Invariantna masa
    axs[2].hist(original_invmass[idx_2000], bins=40, range=invmass_range, alpha=0.3, color='blue', label='Bottom 2000')
    axs[2].hist(original_invmass[idx_1000], bins=40, range=invmass_range, alpha=0.5, color='green', label='Bottom 1000')
    axs[2].hist(original_invmass[idx_500], bins=40, range=invmass_range, alpha=0.8, color='red', label='Bottom 500')
    axs[2].set_title(r'Invariantna masa ($m_{inv}$)')
    axs[2].legend()

    plt.suptitle(naslov_grafa)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, ime_datoteke))
    plt.close()

# Z dodanim minusom funkcija izbere tiste dogodke, ki imajo z_mean^2 najbližje 0
narisi_mase(-z_mean_sq_H, x_H_orig, invmass_H, "Mase delcev - Testni podatki H", '4_mase_testni_Naloga9.pdf')
# ==========================================
# 7. ČRNA ŠKATLA (Naloga 14, 15)
# ==========================================
print("\n--- Gradnja in učenje VAE na podatkih ČRNE ŠKATLE ---")
# Za črno škatlo po navodilih izvedemo ločeno transformacijo in učenje
scaler_bb = QuantileTransformer(output_distribution='uniform')
x_bb_trans = scaler_bb.fit_transform(x_bb_orig)

vae_bb, encoder_bb = build_vae()
vae_bb.fit(x_bb_trans, x_bb_trans, epochs=100, batch_size=1000, verbose=1)

print("Pripravljam grafe mas za črno škatlo...")
z_mean_bb, z_logvar_bb, _ = encoder_bb.predict(x_bb_trans, batch_size=1000)
z_mean_sq_bb = np.square(z_mean_bb).flatten()

# Pri črni škatli navodila svetujejo ožji razpon za invariantno maso (3.6 do 4.0 TeV)
narisi_mase(z_mean_sq_bb, x_bb_orig, invmass_bb, "Mase delcev - Črna škatla", '5_mase_crna_skatla_Naloga15.pdf', invmass_range=(3600, 4000))

print(f"\nKončano! Vsi grafi so shranjeni v mapi '{output_dir}'.")
