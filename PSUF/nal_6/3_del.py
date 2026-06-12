import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('seaborn-v0_8-whitegrid')

# 1. Ustvarimo mapo za slike
os.makedirs('figs_3_del', exist_ok=True)

# ==========================================
# 2. Prilagojen K-means za cilindrično geometrijo (R x S^1)
# ==========================================
def kmeans_cilinder(X, K, max_iter=100, tol=1e-4):
    """
    K-means algoritem, ki upošteva periodične meje za azimutni kot phi.
    X[:, 0] = eta (psevdorapidnost), X[:, 1] = phi (azimutni kot)
    """
    N = len(X)
    # Če imamo manj točk kot je K (kar se lahko zgodi pri močnem brisanju)
    K = min(K, N)
    
    # Naključna inicializacija iz podatkov
    idx = np.random.choice(N, K, replace=False)
    centroidi = X[idx].copy()
    
    for i in range(max_iter):
        # Razdalja v eta
        d_eta = np.abs(X[:, 0:1] - centroidi[:, 0])
        
        # Razdalja v phi (upoštevamo periodičnost 2*pi)
        d_phi = np.abs(X[:, 1:2] - centroidi[:, 1])
        d_phi = np.minimum(d_phi, 2 * np.pi - d_phi)
        
        # Kvadrirana razdalja
        razdalje_sq = d_eta**2 + d_phi**2
        labele = np.argmin(razdalje_sq, axis=1)
        
        novi_centroidi = np.zeros_like(centroidi)
        for k in range(K):
            maska = (labele == k)
            if np.sum(maska) > 0:
                # Običajno povprečje za eta
                novi_centroidi[k, 0] = np.mean(X[maska, 0])
                # Krožno povprečje za phi
                sin_sum = np.sum(np.sin(X[maska, 1]))
                cos_sum = np.sum(np.cos(X[maska, 1]))
                novi_centroidi[k, 1] = np.arctan2(sin_sum, cos_sum)
            else:
                novi_centroidi[k] = centroidi[k]
                
        # Preverjanje konvergence
        d_eta_c = np.abs(novi_centroidi[:, 0] - centroidi[:, 0])
        d_phi_c = np.abs(novi_centroidi[:, 1] - centroidi[:, 1])
        d_phi_c = np.minimum(d_phi_c, 2 * np.pi - d_phi_c)
        if np.max(d_eta_c**2 + d_phi_c**2) < tol**2:
            break
            
        centroidi = novi_centroidi
        
    return centroidi, labele

def izracunaj_maso_higgsa(podatki_dogodka, labele, K):
    """
    Izračuna maso Higgsovega bozona iz dveh curkov z največjim pT.
    Vhodni podatki so oblika: (pT, eta, phi).
    Uporabimo približek E ≈ |p| = pT * cosh(eta).
    """
    pT = podatki_dogodka[:, 0]
    eta = podatki_dogodka[:, 1]
    phi = podatki_dogodka[:, 2]
    
    # 1. Izračunamo skupni pT za vsako gručo
    pT_gruc = np.zeros(K)
    for k in range(K):
        pT_gruc[k] = np.sum(pT[labele == k])
        
    # 2. Poiščemo dve gruči z največjim pT
    najvecji_idx = np.argsort(pT_gruc)[-2:]
    
    # Če imamo samo 1 gručo (zaradi prevelikega brisanja), ne moremo izračunati razpada
    if len(najvecji_idx) < 2:
        return np.nan
        
    # 3. Izračunamo 4-vektor (E, px, py, pz) za vsako od teh dveh gruč
    P_curkov = []
    for idx in najvecji_idx:
        maska = (labele == idx)
        pT_g = pT[maska]
        eta_g = eta[maska]
        phi_g = phi[maska]
        
        px = np.sum(pT_g * np.cos(phi_g))
        py = np.sum(pT_g * np.sin(phi_g))
        pz = np.sum(pT_g * np.sinh(eta_g))
        E  = np.sum(pT_g * np.cosh(eta_g)) # E ≈ |p|
        
        P_curkov.append(np.array([E, px, py, pz]))
        
    # 4. Invariantna masa m = sqrt((E1+E2)^2 - (px1+px2)^2 - (py1+py2)^2 - (pz1+pz2)^2)
    P_tot = P_curkov[0] + P_curkov[1]
    m_sq = P_tot[0]**2 - P_tot[1]**2 - P_tot[2]**2 - P_tot[3]**2
    return np.sqrt(max(0, m_sq))

# ==========================================
# 3. Nalaganje in priprava podatkov
# ==========================================
print("Nalagam podatke...")
# Allow_pickle=True zaradi arrayov različnih dolžin (vsak dogodek ima drugo število delcev)
podatki_vsi = np.load('podatki_PSUF_sklop6/h_bb_sorted.npy', allow_pickle=True)

# Vzamemo prvi dogodek in ga spremenimo v navaden 2D numpy array
prvi_dogodek_tupli = podatki_vsi[0]
prvi_dogodek = np.array([list(d) for d in prvi_dogodek_tupli])

# Podatke sortiramo po pT NARAŠČAJOČE (najmanjši pT so na začetku, da jih lažje odstranjujemo)
prvi_dogodek = prvi_dogodek[np.argsort(prvi_dogodek[:, 0])]

# ==========================================
# 4. Risanje analiz pT (podobno kot kolega - Slika 14)
# ==========================================
pT_vsi = prvi_dogodek[:, 0]
kumulativni_pT = np.cumsum(pT_vsi) / np.sum(pT_vsi)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
ax1.plot(pT_vsi, marker='.', markersize=4, linestyle='-', color='C0')
ax1.set_yscale('log')
ax1.set_title('Transverzalna gibalna količina delcev ($p_T$)')
ax1.set_ylabel('$p_T$ (GeV)')
ax1.set_xlabel('Indeks delca')

ax2.plot(kumulativni_pT, marker='.', markersize=4, linestyle='-', color='C1')
ax2.set_yscale('log')
ax2.set_title('Kumulativni delež celotne $p_T$')
ax2.set_ylabel('Kumulativni delež $p_T$')
ax2.set_xlabel('Indeks delca')

plt.tight_layout()
plt.savefig('figs_3_del/3a_kumulativni_pT.pdf')
plt.close()

# ==========================================
# 5. TEST INFRARDEČE VARNOSTI (IR Safety test)
# ==========================================
stevilo_delcev = len(prvi_dogodek)
# Odstranjevali bomo v korakih (do max 200 odstranjenih, da ohranimo vsaj nekaj pomembnih delcev)
koraki_odstranjevanja = np.arange(0, 201, 5) 
N_ponovitev = 1000 
eksperimentalna_masa = 125.11

vrednosti_K = [10, 20] # Testirali bomo za 10 in 20 gruč
rezultati_testa = {K: {'mean': [], 'std': []} for K in vrednosti_K}

print("\n--- Zagon testa Infrardeče varnosti (IR safety test) ---")
print(f"Skupno število delcev: {stevilo_delcev}. Odstranjujemo do 200 delcev z najmanjšim pT.")

for K in vrednosti_K:
    print(f"Računam za K = {K} ...")
    for n_odstranjenih in koraki_odstranjevanja:
        # Odstranimo 'n_odstranjenih' delcev z najmanjšim pT
        trenutni_dogodek = prvi_dogodek[n_odstranjenih:]
        X_koordinate = trenutni_dogodek[:, 1:3] # Vzamemo samo (eta, phi) za gručenje
        
        mase_ponovitev = []
        for _ in range(N_ponovitev):
            centroidi, labele = kmeans_cilinder(X_koordinate, K)
            masa = izracunaj_maso_higgsa(trenutni_dogodek, labele, K)
            if not np.isnan(masa):
                mase_ponovitev.append(masa)
                
        rezultati_testa[K]['mean'].append(np.mean(mase_ponovitev))
        rezultati_testa[K]['std'].append(np.std(mase_ponovitev))

# ==========================================
# 6. Risanje grafa IR varnosti (Slika 15 od kolega)
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(8, 12))

for idx, K in enumerate(vrednosti_K):
    ax = axes[idx]
    mean_arr = np.array(rezultati_testa[K]['mean'])
    std_arr = np.array(rezultati_testa[K]['std'])
    
    # Rumeni trak = interval zaupanja (1 std. odklon) za n=0 odstranjenih (vsi delci)
    osnovni_mean = mean_arr[0]
    osnovni_std = std_arr[0]
    ax.axhspan(osnovni_mean - osnovni_std, osnovni_mean + osnovni_std, 
               color='lightsteelblue', alpha=0.3, label='K-means: z vsemi delci')
    
    # Idealna eksperimentalna vrednost
    ax.axhline(eksperimentalna_masa, color='black', linestyle='--', linewidth=1, label=f'$m_H^{{exp}} = {eksperimentalna_masa}$ GeV')
    
    # Rezultati z napakami
    ax.errorbar(koraki_odstranjevanja, mean_arr, yerr=std_arr, fmt='.-', color='C0', 
                elinewidth=1, markersize=6, label='z odstranjevanjem delcev')
    
    ax.set_title(f'Infrardeča varnost algoritma K-means: K = {K}')
    ax.set_ylabel('$m_H$ (GeV)')
    ax.set_xlabel('št. odstranjenih delcev')
    ax.legend(loc='lower left' if K == 20 else 'upper left')

plt.tight_layout()
plt.savefig('figs_3_del/3b_ir_varnost.pdf')
plt.close()

print("\nKoda se je uspešno izvedla. Slike za 3. del (infrardeča varnost in pT analiza) so shranjene v mapi 'figs_3_del/'.")