import numpy as np
import matplotlib.pyplot as plt
import os
import fastjet as fj

# Poenoten stil za grafe
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Ustvarimo mapo za slike
os.makedirs('figs_7_del', exist_ok=True)

# ==========================================
# 2. Nalaganje podatkov
# ==========================================
print("Nalagam 10.000 dogodkov za 7. nalogo...")
podatki_vsi = np.load('podatki_PSUF_sklop6/h_bb_sorted.npy', allow_pickle=True)

mase_vseh_dogodkov = []

# ==========================================
# 3. Zagon algoritma na vseh dogodkih
# ==========================================
print("Začenjam gručenje za vseh 10.000 dogodkov (fastjet, k_t, R=0.6) ...")

# Pripravimo definicijo curkov zunaj zanke (je ista za vse dogodke)
jet_def = fj.JetDefinition(fj.kt_algorithm, 0.6)

# Iteriramo čez vse dogodke
for i, dogodek in enumerate(podatki_vsi):
    # Priprava dogodka za fastjet
    dogodek_fastjet = [fj.PtYPhiM(d[0], d[1], d[2], d[3] if len(d) > 3 else 0.0) for d in dogodek]
    
    # Gručenje
    cs = fj.ClusterSequence(dogodek_fastjet, jet_def)
    incl_jets = fj.sorted_by_pt(cs.inclusive_jets())
    
    # Če sta vsaj dva curka, izračunamo maso Higgsa
    if len(incl_jets) >= 2:
        j1, j2 = incl_jets[0], incl_jets[1]
        
        # Izračun invariantne mase iz 4-količin
        m_sq = (j1.e() + j2.e())**2 - (j1.px() + j2.px())**2 - (j1.py() + j2.py())**2 - (j1.pz() + j2.pz())**2
        if m_sq >= 0:
            mase_vseh_dogodkov.append(np.sqrt(m_sq))

mase_vseh_dogodkov = np.array(mase_vseh_dogodkov)

# ==========================================
# 4. Statistična obdelava (Določitev mase in napake)
# ==========================================
# a) Naivno povprečje celotnega spektra (ki ga povlečejo anomalije)
naivno_povprecje = np.mean(mase_vseh_dogodkov)
naivni_std = np.std(mase_vseh_dogodkov)

# b) Pametnejša določitev: Iskanje vrha (mode) histograma
# Ustvarimo histogram s 100 bin-i med 0 in 300 GeV
counts, bins = np.histogram(mase_vseh_dogodkov, bins=100, range=(0, 300))
max_bin_idx = np.argmax(counts)
vrh_mase = (bins[max_bin_idx] + bins[max_bin_idx+1]) / 2

# Za napako vzamemo standardni odklon samo tistih dogodkov, ki so v jedru vrha (npr. vrh ± 20 GeV)
okno_spodaj = vrh_mase - 20
okno_zgoraj = vrh_mase + 20
jedro_mas = mase_vseh_dogodkov[(mase_vseh_dogodkov >= okno_spodaj) & (mase_vseh_dogodkov <= okno_zgoraj)]

koncna_masa = np.mean(jedro_mas)
koncna_napaka = np.std(jedro_mas)

print("\n--- REZULTATI OBDELAVE ---")
print(f"Uspešno rekonstruiranih dogodkov: {len(mase_vseh_dogodkov)} / {len(podatki_vsi)}")
print(f"Naivno povprečje vseh mas: {naivno_povprecje:.2f} ± {naivni_std:.2f} GeV (precenjeno zaradi asimetričnega repa!)")
print(f"Položaj vrha histograma (mode): ~{vrh_mase:.2f} GeV")
print(f"Končna ocena mase (iz robustnega jedra): {koncna_masa:.2f} ± {koncna_napaka:.2f} GeV")
print(f"Eksperimentalna masa: 125.11 GeV")

# ==========================================
# 5. Izris in shranjevanje spektra (Histogram)
# ==========================================

# SLIKA 1: Linearni spekter
plt.figure(figsize=(9, 6))
# Uporabimo privzeto modro barvo iz C0 z malo transparence
plt.hist(mase_vseh_dogodkov, bins=150, range=(0, 550), color='C0', alpha=0.8, edgecolor='black', linewidth=0.5)
plt.axvline(naivno_povprecje, color='C3', linestyle='-', linewidth=2, label=f'$\\overline{{m_H}} = {naivno_povprecje:.1f}$ GeV (povprečje)')
plt.axvline(125.11, color='black', linestyle='--', linewidth=2, label=r'$m_H^{exp} = 125.11$ GeV')

plt.title('Histogram $m_H$ iz algoritma fastjet $k_t$, $R=0.6$')
plt.xlabel('$m_H$ [GeV]')
plt.ylabel('Število dogodkov')
plt.legend()
plt.tight_layout()
plt.savefig('figs_7_del/7_histogram_linearen.pdf')
plt.close()

# SLIKA 2: Logaritemski spekter
plt.figure(figsize=(9, 6))
plt.hist(mase_vseh_dogodkov, bins=300, range=(0, 3500), color='C0', alpha=0.8, edgecolor='none')
plt.axvline(naivno_povprecje, color='C3', linestyle='-', linewidth=2, label=f'$\\overline{{m_H}} = {naivno_povprecje:.1f}$ GeV (povprečje)')
plt.axvline(125.11, color='black', linestyle='--', linewidth=2, label=r'$m_H^{exp} = 125.11$ GeV')

plt.yscale('log')
plt.title('Histogram $m_H$ iz algoritma fastjet $k_t$, $R=0.6$ (logaritemska skala)')
plt.xlabel('$m_H$ [GeV]')
plt.ylabel('Število dogodkov')
plt.legend()
plt.tight_layout()
plt.savefig('figs_7_del/7_histogram_log.pdf')
plt.close()

print("\nIzris in shranjevanje uspešno. Slike so v mapi 'figs_7_del/'.")