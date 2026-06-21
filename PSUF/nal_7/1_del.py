import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# 0. Ustvarimo mapo za grafe, če ta še ne obstaja
output_dir = "figs_1_del"
os.makedirs(output_dir, exist_ok=True)

# 1. Nalaganje in priprava podatkov
print("Nalagam MNIST podatke (to lahko traja kakšno minuto)...")
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

# Normalizacija na interval [0, 1]
x = X / 255.0
y = np.array(y, dtype=int)
print(f"Podatki naloženi! Velikost: {x.shape}")

# 2. Inicializacija parametrov za binarno mešanico
np.random.seed(42) # Za ponovljivost rezultatov
n_clusters = 10
n_samples, n_features = x.shape

# Deleži gruč (pi) so na začetku enakomerno porazdeljeni
pi = np.ones(n_clusters) / n_clusters

# Začetni vektorji mu so naključno izbrani iz enakomerne porazdelitve. 
# Izbiramo jih nekoliko stran od robov 0 in 1, da preprečimo log(0) pri prvem koraku.
mu = np.random.uniform(0.25, 0.75, size=(n_clusters, n_features))

n_iter = 100
history_delta_mu = []

print("Začenjam učenje binarne mešanice (EM algoritem)...")
for i in range(n_iter):
    # Da preprečimo prelivanje napak ali deljenje z 0 (log(0) ni definiran), 
    # vrednosti mu "obstrižemo" (clip) malenkost nad 0 in pod 1.
    mu_clipped = np.clip(mu, 1e-10, 1 - 1e-10)
    
    # --- E-KORAK ---
    # Izračunamo logaritem verjetnosti P(x|mu) za vse slike in vse gruče naenkrat 
    # P(x|mu) = produkt( mu_i^x_i * (1-mu_i)^(1-x_i) ) --> V log prostoru postane vsota!
    log_p_x_given_mu = np.dot(x, np.log(mu_clipped).T) + np.dot(1 - x, np.log(1 - mu_clipped).T)
    
    # Števec za aposteriorno verjetnost v log prostoru: log( P(x|mu) * pi )
    A = log_p_x_given_mu + np.log(pi)
    
    # Uporaba trika iz navodil: e^{A_j + B} / sum(e^{A_j + B}) za numerično stabilnost
    B = -np.max(A, axis=1, keepdims=True)
    numerator = np.exp(A + B)
    denominator = np.sum(numerator, axis=1, keepdims=True)
    
    # gamma = P(x_l | j) - Odgovornost (responsibility), da slika x_l pripada gruči j
    gamma = numerator / denominator 
    
    # --- M-KORAK ---
    N_j = np.sum(gamma, axis=0) # Skupno število dodeljenih slik vsaki gruči
    
    # Nove vrednosti povprečij (mu) in deležev (pi)
    mu_new = np.dot(gamma.T, x) / N_j[:, None]
    pi_new = N_j / n_samples
    
    # Spremljanje konvergence (koliko se parametra mu spremenijo)
    delta_mu = np.mean(np.abs(mu_new - mu))
    history_delta_mu.append(delta_mu)
    
    mu = mu_new
    pi = pi_new
    
    if (i + 1) % 10 == 0:
        print(f"Iteracija {i + 1}/{n_iter} zaključena. Povprečna sprememba \u0394\u03bc: {delta_mu:.6f}")

print("Učenje zaključeno!")

# 3. GRAFI IN VIZUALIZACIJA

# GRAF 1: Spreminjanje parametrov po korakih (Konvergenca)
plt.figure(figsize=(8, 5))
plt.plot(history_delta_mu, 'b.-')
plt.yscale('log')
plt.xlabel('Korak')
plt.ylabel(r'$\Delta \mu$')
plt.title('Spreminjanje parametrov po korakih')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.savefig(os.path.join(output_dir, '1_konvergenca_parametrov.pdf'), bbox_inches='tight')
plt.close()

# GRAF 2: Prikaz naučenih "povprečnih" števk (\mu^j)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for j, ax in enumerate(axes.flatten()):
    # Sliko \mu^j transformiramo nazaj v mrežo 28x28
    ax.imshow(mu[j].reshape(28, 28), cmap='gray')
    ax.axis('off')
    ax.set_title(f'Gruča {j+1}')
plt.suptitle("Naučeni vektorji $\mu^j$ za binarne mešanice", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_naucene_slike_mu.pdf'), bbox_inches='tight')
plt.close()

# GRAF 3: Generiranje novih števk (uporaba np.random.binomial)
# Po navodilih pogledamo, kaj se zgodi pri žrebanju za različne n (1, 3, 9)
clusters_to_plot = [0, 1, 2, 3, 4] # Izberemo prvih 5 gruč za vzorec
n_values = [1, 3, 9]

fig, axes = plt.subplots(len(clusters_to_plot), len(n_values), figsize=(3*len(n_values), 3*len(clusters_to_plot)))

for row_idx, j in enumerate(clusters_to_plot):
    for col_idx, n_val in enumerate(n_values):
        ax = axes[row_idx, col_idx]
        # Žrebanje iz binomske porazdelitve (generacija pik)
        x_new = np.random.binomial(n=n_val, p=mu[j])
        ax.imshow(x_new.reshape((28, 28)), cmap='gray')
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(f'Binomski n={n_val}')
        if col_idx == 0:
            ax.text(-5, 14, f'Gruča {j+1}', va='center', ha='right', fontsize=12)

plt.suptitle("Generirane (izžrebane) števke s povečevanjem parametra $n$", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_generirane_stevke.pdf'), bbox_inches='tight')
plt.close()

print(f"Vsi grafi 1. dela so bili uspešno shranjeni v mapo '{output_dir}/'!")