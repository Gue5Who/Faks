import numpy as np
import matplotlib.pyplot as plt
import os
import time
# pyrefly: ignore [missing-import]
from pyjet import DTYPE_PTEPM, cluster

# 1. Ustvarimo mapo za slike
os.makedirs('figs_5+6_del', exist_ok=True)

# ==========================================
# 2. Pomožne funkcije (Tvoj algoritem)
# ==========================================
def izracunaj_maso_higgsa_iz_pravih_curkov(curki):
    """Izračun mase z uporabo približka E ≈ |p| = pT * cosh(eta)."""
    if len(curki) < 2:
        return np.nan
    
    curki = curki[np.argsort(curki[:, 0])[::-1]] # Sortiramo padajoče po pT
    c1, c2 = curki[0], curki[1]
    
    def v4(curek):
        pt, eta, phi = curek[:3]
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        E = pt * np.cosh(eta)
        return np.array([E, px, py, pz])
        
    p1 = v4(c1)
    p2 = v4(c2)
    P_tot = p1 + p2
    m_sq = P_tot[0]**2 - P_tot[1]**2 - P_tot[2]**2 - P_tot[3]**2
    return np.sqrt(max(0, m_sq))

def hierarhicno_grucenje(podatki, p=1, R=0.6):
    """Tvoja implementacija algoritma kt."""
    proto_curki = list(podatki.copy()[:, :3])
    pravi_curki = []
    
    while len(proto_curki) > 0:
        pT = np.array([c[0] for c in proto_curki])
        eta = np.array([c[1] for c in proto_curki])
        phi = np.array([c[2] for c in proto_curki])
        
        d_i = pT**(2 * p)
        pT_2p = pT**(2 * p)
        min_pT_2p = np.minimum(pT_2p[:, None], pT_2p[None, :])
        
        d_eta = eta[:, None] - eta[None, :]
        d_phi = phi[:, None] - phi[None, :]
        d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi
        
        d_ij = min_pT_2p * (d_eta**2 + d_phi**2) / (R**2)
        np.fill_diagonal(d_ij, np.inf)
        
        min_di_idx = np.argmin(d_i)
        min_di_val = d_i[min_di_idx]
        
        min_dij_flat_idx = np.argmin(d_ij)
        min_dij_val = d_ij.flat[min_dij_flat_idx]
        
        if min_dij_val < min_di_val:
            i, j = np.unravel_index(min_dij_flat_idx, d_ij.shape)
            if i > j: i, j = j, i
                
            pt_i, eta_i, phi_i = proto_curki[i]
            pt_j, eta_j, phi_j = proto_curki[j]
            
            pt_k = pt_i + pt_j
            eta_k = (pt_i * eta_i + pt_j * eta_j) / pt_k
            
            dphi = phi_i - phi_j
            if dphi > np.pi: phi_i -= 2 * np.pi
            elif dphi < -np.pi: phi_i += 2 * np.pi
                
            phi_k = (pt_i * phi_i + pt_j * phi_j) / pt_k
            phi_k = (phi_k + np.pi) % (2 * np.pi) - np.pi
            
            novi_protocurek = np.array([pt_k, eta_k, phi_k])
            
            proto_curki.pop(j)
            proto_curki.pop(i)
            proto_curki.append(novi_protocurek)
        else:
            pravi_curki.append(proto_curki.pop(min_di_idx))
            
    return np.array(pravi_curki)

# ==========================================
# 3. Priprava podatkov in iteracija
# ==========================================
print("Nalagam podatke in izbiram 1. dogodek...")
podatki_vsi = np.load('podatki_PSUF_sklop6/h_bb_sorted.npy', allow_pickle=True)

# Prvi dogodek kot seznam tuplov, sortiran NARAŠČAJOČE po pT
prvi_dogodek_sorted = sorted(podatki_vsi[0], key=lambda x: x[0])

# Parametri za iteracijo
koraki_odstranjevanja = np.arange(0, 201, 10)
R_vrednosti = [0.4, 0.5, 0.6, 0.7, 1.0] # Testirani R-ji
p_param = 1 # algoritem k_t

# Slovarji za shranjevanje rezultatov za vse R
mase_moj = {R: [] for R in R_vrednosti}
mase_pyjet = {R: [] for R in R_vrednosti}
casi_moj = {R: [] for R in R_vrednosti}
casi_pyjet = {R: [] for R in R_vrednosti}

print("\n--- Zagon analize IR varnosti in časovne zahtevnosti za vse R ---")
print("Opomba: Tvoja implementacija se bo izvajala okoli 1-2 minuti...")

for n_odstranjenih in koraki_odstranjevanja:
    trenutni_dogodek = prvi_dogodek_sorted[n_odstranjenih:]
    dogodek_moj_np = np.array([list(d) for d in trenutni_dogodek])
    dogodek_pyjet = np.array(trenutni_dogodek, dtype=DTYPE_PTEPM)
    
    for R in R_vrednosti:
        # ---------------------------
        # A) Tvoj algoritem
        # ---------------------------
        t0 = time.time()
        curki_moj = hierarhicno_grucenje(dogodek_moj_np, p=p_param, R=R)
        t1 = time.time()
        mase_moj[R].append(izracunaj_maso_higgsa_iz_pravih_curkov(curki_moj))
        casi_moj[R].append(t1 - t0)
        
        # ---------------------------
        # B) PyJet algoritem
        # ---------------------------
        zagoni_pyjet = 50 # Za zanesljivo meritev časa pri ultra-hitrem pyjet-u
        t0_pyjet = time.time()
        for _ in range(zagoni_pyjet):
            sequence = cluster(dogodek_pyjet, ep=False, R=R, p=p_param)
            incl_jets = sequence.inclusive_jets()
        t1_pyjet = time.time()
        casi_pyjet[R].append((t1_pyjet - t0_pyjet) / zagoni_pyjet)
        
        if len(incl_jets) >= 2:
            incl_jets.sort(key=lambda j: j.pt, reverse=True)
            j1, j2 = incl_jets[0], incl_jets[1]
            m_sq = (j1.e + j2.e)**2 - (j1.px + j2.px)**2 - (j1.py + j2.py)**2 - (j1.pz + j2.pz)**2
            mase_pyjet[R].append(np.sqrt(max(0, m_sq)))
        else:
            mase_pyjet[R].append(np.nan)

# ==========================================
# 4. Izris in shranjevanje grafov
# ==========================================
barve = {0.4: 'tab:blue', 0.5: 'tab:orange', 0.6: 'tab:green', 0.7: 'tab:red', 1.0: 'tab:purple'}

# --- SLIKA 1: IR varnost samo tvoj algoritem ---
plt.figure(figsize=(8, 6))
for R in R_vrednosti:
    plt.plot(koraki_odstranjevanja, mase_moj[R], label=f'$R={R}$', marker='.', color=barve[R])

plt.axhline(125.11, color='blue', linestyle='-', linewidth=4, alpha=0.2, label='$m_H^{exp} = 125.11 \pm 0.11$ GeV')
plt.title(f'Infrardeča varnost mojega algoritma $k_t$')
plt.xlabel('št. odstranjenih delcev')
plt.ylabel('$m_H$ (GeV)')
plt.legend()
plt.tight_layout()
plt.savefig('figs_5+6_del/5_IR_varnost_moj.png')
plt.close()


# --- SLIKA 2: IR varnost samo PyJet algoritem ---
plt.figure(figsize=(8, 6))
for R in R_vrednosti:
    plt.plot(koraki_odstranjevanja, mase_pyjet[R], label=f'$R={R}$', marker='.', color=barve[R])

plt.axhline(125.11, color='blue', linestyle='-', linewidth=4, alpha=0.2, label='$m_H^{exp} = 125.11 \pm 0.11$ GeV')
plt.title(f'Infrardeča varnost pyjet algoritma $k_t$')
plt.xlabel('št. odstranjenih delcev')
plt.ylabel('$m_H$ (GeV)')
plt.legend()
plt.tight_layout()
plt.savefig('figs_5+6_del/5_IR_varnost_pyjet.png')
plt.close()


# --- SLIKA 3: Primerjava obeh (Masa) ---
plt.figure(figsize=(9, 7))
for R in R_vrednosti:
    plt.plot(koraki_odstranjevanja, mase_moj[R], label=f'$R={R}$, moj', marker='.', linestyle='-', color=barve[R])
    plt.plot(koraki_odstranjevanja, mase_pyjet[R], label=f'$R={R}$, pyjet', marker='', linestyle='--', color=barve[R], alpha=0.7)

plt.axhline(125.11, color='blue', linestyle='-', linewidth=4, alpha=0.2, label='$m_H^{exp} = 125.11 \pm 0.11$ GeV')
plt.title(f'IR varnost mojega in pyjet algoritma $k_t$')
plt.xlabel('št. odstranjenih delcev')
plt.ylabel('$m_H$ (GeV)')
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig('figs_5+6_del/5_IR_varnost_primerjava.png')
plt.close()


# --- SLIKA 4: Primerjava časov izvajanja za vse R (samo logaritemska skala) ---
plt.figure(figsize=(10, 7))
for R in R_vrednosti:
    plt.plot(koraki_odstranjevanja, casi_moj[R], label=f'$R={R}$, moj', marker='.', linestyle='-', color=barve[R])
    plt.plot(koraki_odstranjevanja, casi_pyjet[R], label=f'$R={R}$, pyjet', marker='x', linestyle='--', color=barve[R], alpha=0.8)

plt.title('Časovna zahtevnost mojega in pyjet algoritma $k_t$ (logaritemska skala)')
plt.xlabel('št. odstranjenih delcev (z najmanjšim $p_T$)')
plt.ylabel('Čas izvajanja t (s)')
plt.yscale('log')
plt.grid(True, which="both", linestyle='--', alpha=0.4)

# Legendo premaknemo izven grafa oz. jo postavimo v dva stolpca, da ne prekriva črt
plt.legend(ncol=2, loc='center left')

plt.tight_layout()
plt.savefig('figs_5+6_del/6_cas_izvajanja_vsi_R.png', bbox_inches='tight')
plt.close()

print("\nIzračuni so končani. Vse 4 slike so shranjene v mapo 'figs_5+6_del/'.")