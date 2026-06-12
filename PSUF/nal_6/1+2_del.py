import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering

plt.style.use('seaborn-v0_8-whitegrid')

# 1. Ustvarimo mapo za slike, če še ne obstaja
os.makedirs('figs_1+2_del', exist_ok=True)

# 2. Lastna implementacija K-means algoritma
def moj_kmeans(X, K, max_iter=100, tol=1e-4):
    """
    Algoritem K-means za iskanje gruč v podatkih X.
    Vrne: centroidi, labele, vztrajnost (inertia - vsota kvadratov razdalj)
    """
    np.random.seed(42)  # Za ponovljivost rezultatov
    # Naključna izbira začetnih centroidov neposredno izmed podatkov
    idx = np.random.choice(len(X), K, replace=False)
    centroidi = X[idx]
    
    for i in range(max_iter):
        # Izračun Evklidske razdalje od vsake točke do vsakega centroida
        razdalje = np.linalg.norm(X[:, np.newaxis] - centroidi, axis=2)
        
        # Določitev pripadajoče gruče (indeks centroida z minimalno razdaljo)
        labele = np.argmin(razdalje, axis=1)
        
        # Posodobitev vrednosti centroidov (povprečje točk v posamezni gruči)
        # Če kaka gruča ostane prazna, obdržimo stari centroid, da ne pride do napake (deljenja z 0)
        novi_centroidi = np.array([X[labele == k].mean(axis=0) if np.sum(labele == k) > 0 
                                   else centroidi[k] for k in range(K)])
        
        # Preverjanje konvergence (če se centroidi ne premikajo več bistveno)
        if np.all(np.abs(novi_centroidi - centroidi) < tol):
            break
            
        centroidi = novi_centroidi
        
    # Izračun vztrajnosti (Inertia - WCSS: Within-Cluster Sum of Squares)
    vztrajnost = 0
    for k in range(K):
        točke_v_gruči = X[labele == k]
        if len(točke_v_gruči) > 0:
            # Kvadratna razdalja do centroida
            vztrajnost += np.sum(np.linalg.norm(točke_v_gruči - centroidi[k], axis=1)**2)
            
    return centroidi, labele, vztrajnost

# Naložimo sintetične podatke
podatki = np.load('podatki_PSUF_sklop6/gauss.npy')

# ==========================================
# DEL 1A: Metoda komolca (Elbow Method) za različne K (od 1 do 12)
# ==========================================
vrednosti_K_komolec = range(1, 13)
vztrajnosti = []

print("--- Računanje Metode komolca za K od 1 do 12 ---")
for K in vrednosti_K_komolec:
    _, _, vztrajnost = moj_kmeans(podatki, K)
    vztrajnosti.append(vztrajnost)

# Izris in shranjevanje grafa Metode komolca
plt.figure(figsize=(8, 6))
plt.plot(vrednosti_K_komolec, vztrajnosti, marker='o', linestyle='-', color='C0')
plt.title('Elbow Method')
plt.xlabel('K')
plt.ylabel('Inertia - WCSS')
plt.xticks(vrednosti_K_komolec)
plt.tight_layout()
plt.savefig('figs_1+2_del/1_komolec_K1_do_12.pdf')
plt.close()

# ==========================================
# DEL 1B: Moj K-means za različne izbrane K 
# ==========================================
izbrani_K = [2, 3, 4, 5, 6]

print("\n--- Rezultati gručenja za izbrane vrednosti K ---")
for K in izbrani_K:
    centroidi, labele, _ = moj_kmeans(podatki, K)
    
    plt.figure(figsize=(8, 6))
    for k in range(K):
        plt.scatter(podatki[labele == k, 0], podatki[labele == k, 1], label=f'G{k+1}', s=20)
    plt.scatter(centroidi[:, 0], centroidi[:, 1], c='black', marker='x', s=100, linewidths=2, label='Centroidi')
    plt.title(f'Lastna implementacija K-means (K = {K})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(fontsize='large')
    plt.tight_layout()
    # Datoteka ima K v imenu (npr. figs_1+2_del/1_moj_kmeans_K2.pdf)
    plt.savefig(f'figs_1+2_del/1_moj_kmeans_K{K}.pdf')
    plt.close()
    
    # Izpis statistike zgolj za optimalni primer K=2, kot zahtevajo navodila
    if K == 2:
        print("\n=> Statistika za optimalni primer (K=2) iz mojega algoritma:")
        for k in range(K):
            tocke_gruce = podatki[labele == k]
            povprecje = np.mean(tocke_gruce, axis=0)
            std = np.std(tocke_gruce, axis=0)
            print(f"Gruča {k+1}: Povprečje = {povprecje.round(3)}, Standardni odklon = {std.round(3)}")


# ==========================================
# DEL 2: Gaussian Mixture Model (GMM) - preizkus za K=2
# ==========================================
K_opt = 2
gmm = GaussianMixture(n_components=K_opt, random_state=42)
labele_gmm = gmm.fit_predict(podatki)

print("\n--- Rezultati Gaussian Mixture Model (GMM) za K=2 ---")
for k in range(K_opt):
    print(f"Gruča {k+1}: Povprečje = {gmm.means_[k].round(3)}")
    std_gmm = np.sqrt(np.diag(gmm.covariances_[k]))
    print(f"         Standardni odklon = {std_gmm.round(3)}")

plt.figure(figsize=(8, 6))
for k in range(K_opt):
    plt.scatter(podatki[labele_gmm == k, 0], podatki[labele_gmm == k, 1], label=f'G{k+1}', s=20)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='black', marker='x', s=100, linewidths=2, label='Središča GMM')
plt.title(f'Gručenje z GMM (K = {K_opt})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(fontsize='large')
plt.tight_layout()
plt.savefig(f'figs_1+2_del/2_gmm_K{K_opt}.pdf')
plt.close()

# ==========================================
# DEL 3: Trije dodatni algoritmi iz scikit-learn (preizkus za K=2)
# ==========================================
print("\n--- Računanje treh dodatnih algoritmov iz scikit-learn (za K=2) ---")

# a) scikit-learn KMeans
kmeans_sk = KMeans(n_clusters=K_opt, random_state=42, n_init='auto')
labele_sk_kmeans = kmeans_sk.fit_predict(podatki)

plt.figure(figsize=(8, 6))
for k in range(K_opt):
    plt.scatter(podatki[labele_sk_kmeans == k, 0], podatki[labele_sk_kmeans == k, 1], label=f'G{k+1}', s=20)
plt.title(f'Gručenje s scikit-learn KMeans (K = {K_opt})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(fontsize='large')
plt.tight_layout()
plt.savefig(f'figs_1+2_del/3_sklearn_kmeans_K{K_opt}.pdf')
plt.close()

# b) Agglomerative Clustering (Hierarhično)
agg = AgglomerativeClustering(n_clusters=K_opt)
labele_agg = agg.fit_predict(podatki)

plt.figure(figsize=(8, 6))
for k in range(K_opt):
    plt.scatter(podatki[labele_agg == k, 0], podatki[labele_agg == k, 1], label=f'G{k+1}', s=20)
plt.title(f'Gručenje z Agglomerative Clustering (K = {K_opt})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(fontsize='large')
plt.tight_layout()
plt.savefig(f'figs_1+2_del/4_sklearn_agglomerative_K{K_opt}.pdf')
plt.close()

# c) Spectral Clustering
spectral = SpectralClustering(n_clusters=K_opt, affinity='nearest_neighbors', random_state=42)
labele_spec = spectral.fit_predict(podatki)

plt.figure(figsize=(8, 6))
barve = ['C1', 'C0']
for k in range(K_opt):
    plt.scatter(podatki[labele_spec == k, 0], podatki[labele_spec == k, 1], color=barve[k], label=f'G{k+1}', s=20)
plt.title(f'Gručenje s Spectral Clustering (K = {K_opt})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(fontsize='large')
plt.tight_layout()
plt.savefig(f'figs_1+2_del/5_sklearn_spectral_K{K_opt}.pdf')
plt.close()

print("\nIzris in shranjevanje vseh slik uspešno zaključeno. Datoteke se nahajajo v mapi 'figs_1+2_del/'.")