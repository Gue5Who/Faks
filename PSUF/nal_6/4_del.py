import numpy as np

# ==========================================
# 1. Funkcija za izračun mase iz pravih curkov
# ==========================================
def izracunaj_maso_higgsa_iz_pravih_curkov(curki):
    """
    Izračuna maso Higgsovega bozona iz seznama končnih (pravih) curkov.
    Vzame 2 curka z največjim pT in izračuna njuno invariantno maso.
    """
    if len(curki) < 2:
        return np.nan
    
    # Sortiramo curke po pT padajoče (največji pT na začetku)
    curki = curki[np.argsort(curki[:, 0])[::-1]]
    
    # Izberemo dva z največjim pT
    c1, c2 = curki[0], curki[1]
    
    def v4(curek):
        pt, eta, phi = curek
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        E = pt * np.cosh(eta) # Uporabimo približek E \approx |p|
        return np.array([E, px, py, pz])
        
    p1 = v4(c1)
    p2 = v4(c2)
    P_tot = p1 + p2
    
    # Invariantna masa m = sqrt(E^2 - px^2 - py^2 - pz^2)
    m_sq = P_tot[0]**2 - P_tot[1]**2 - P_tot[2]**2 - P_tot[3]**2
    return np.sqrt(max(0, m_sq))

# ==========================================
# 2. Implementacija hierarhičnega algoritma
# ==========================================
def hierarhicno_grucenje(podatki, p, R):
    """
    Implementacija algoritma hierarhičnega gručenja v 5 korakih iz navodil.
    podatki: numpy array oblike (N, 3), kjer so stolpci (pT, eta, phi).
    p: parameter algoritma (-1 za anti-kt, 0 za Cambridge-Aachen, 1 za kt).
    R: parameter radija.
    Vrne: seznam končnih (pravih) curkov.
    """
    # Začnemo s seznamom vseh začetnih delcev (protocurkov)
    proto_curki = list(podatki.copy()[:, :3]) # vzamemo le pT, eta, phi
    pravi_curki = []
    
    while len(proto_curki) > 0:
        N = len(proto_curki)
        
        # Ekstrahiramo lastnosti v numpy polja za hitrejši izračun
        pT = np.array([c[0] for c in proto_curki])
        eta = np.array([c[1] for c in proto_curki])
        phi = np.array([c[2] for c in proto_curki])
        
        # 1. korak: Izračun d_i
        d_i = pT**(2 * p)
        
        # 1. korak: Izračun d_ij
        pT_2p = pT**(2 * p)
        # Matrika minimalnih vrednosti pT_i in pT_j
        min_pT_2p = np.minimum(pT_2p[:, None], pT_2p[None, :])
        
        d_eta = eta[:, None] - eta[None, :]
        d_phi = phi[:, None] - phi[None, :]
        # Periodični robni pogoji za kot phi (razdalja je mod 2*pi)
        d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi
        
        d_ij = min_pT_2p * (d_eta**2 + d_phi**2) / (R**2)
        
        # Na diagonalo damo neskončnost, da se delec ne združuje sam s seboj
        np.fill_diagonal(d_ij, np.inf)
        
        # 2. korak: Poiščemo minimalno razdaljo d_min
        min_di_idx = np.argmin(d_i)
        min_di_val = d_i[min_di_idx]
        
        min_dij_flat_idx = np.argmin(d_ij)
        min_dij_val = d_ij.flat[min_dij_flat_idx]
        
        # Preverimo, kateri d_min je absolutni minimum
        if min_dij_val < min_di_val:
            # 3. korak: d_min je d_ij -> Združimo protocurka i in j v novega k
            i, j = np.unravel_index(min_dij_flat_idx, d_ij.shape)
            
            # Da bomo lahko varno odstranili elementa iz seznama (od zadaj naprej)
            if i > j:
                i, j = j, i
                
            pt_i, eta_i, phi_i = proto_curki[i]
            pt_j, eta_j, phi_j = proto_curki[j]
            
            # Enačbe za združevanje (Enačba 3 v navodilih)
            pt_k = pt_i + pt_j
            eta_k = (pt_i * eta_i + pt_j * eta_j) / pt_k
            
            # Paziti moramo pri povprečenju phi zaradi periodičnosti
            dphi = phi_i - phi_j
            if dphi > np.pi:
                phi_i -= 2 * np.pi
            elif dphi < -np.pi:
                phi_i += 2 * np.pi
                
            phi_k = (pt_i * phi_i + pt_j * phi_j) / pt_k
            phi_k = (phi_k + np.pi) % (2 * np.pi) - np.pi # Zapakiramo nazaj v [-pi, pi]
            
            novi_protocurek = np.array([pt_k, eta_k, phi_k])
            
            # 4. korak (delno): Odstranimo stara in dodamo novega 
            # Odstranimo večji indeks najprej, da se manjši indeks ne premakne!
            proto_curki.pop(j)
            proto_curki.pop(i)
            proto_curki.append(novi_protocurek)
            
        else:
            # 4. korak: d_min je d_i -> Protocurek i proglasimo za pravi curek
            pravi_curki.append(proto_curki.pop(min_di_idx))
            
    return np.array(pravi_curki)

# ==========================================
# 3. Nalaganje podatkov in Zagon
# ==========================================
print("Nalagam podatke za 4. nalogo...")
podatki_vsi = np.load('podatki_PSUF_sklop6/h_bb_sorted.npy', allow_pickle=True)
prvi_dogodek = np.array([list(d) for d in podatki_vsi[0]])

algoritmi = {
    -1: 'anti-k_t',
     0: 'Cambridge-Aachen',
     1: 'k_t'
}

print("\n--- Izračun mase Higgsovega bozona (R = 0.6) ---")
for p, ime in algoritmi.items():
    najdeni_curki = hierarhicno_grucenje(prvi_dogodek, p=p, R=0.6)
    masa = izracunaj_maso_higgsa_iz_pravih_curkov(najdeni_curki)
    print(f"Algoritem {ime:<18} (p = {p:2d}): m_H = {masa:.2f} GeV (Št. najdenih curkov: {len(najdeni_curki)})")


print("\n--- Analiza vpliva parametra R (testiramo z anti-k_t, p = -1) ---")
radiji = [0.2, 0.6, 1.0, 1.5]
for R_test in radiji:
    najdeni_curki = hierarhicno_grucenje(prvi_dogodek, p=-1, R=R_test)
    masa = izracunaj_maso_higgsa_iz_pravih_curkov(najdeni_curki)
    print(f"Radij R = {R_test:.1f}  ->  m_H = {masa:.2f} GeV  (Št. curkov: {len(najdeni_curki)})")