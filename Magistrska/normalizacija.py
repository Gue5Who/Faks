"""
Normalizacija kontinuuma: nova metoda (b, c) proti stari metodi (točke modela).

Vhodni FITS (_c.fits) so že pred-obdelani na telesku, fluks v oknu Na D leži
okoli 1. Zato ni nobene prednormalizacije: vse se dela na SUROVIH podatkih.

NOVA METODA (b, c):
    sme_rv_vsi.py je na surovem fluksu fitiral
        y_surov ~= model + b + c * (x - x_sredina)
    (kovinske črte, maskiran Na D) in shranil b_offset in c_slope v tabelo.
    Tam, kjer je model na kontinuumu (model = 1), je kontinuum opazovanega
    spektra
        C_bc(x) = 1 + b + c * (x - x_sredina)
    Normalizacija (nastavitev NACIN_NORM_BC):
        "deljenje":    y_norm = y_surov / C_bc
        "odstevanje":  y_norm = y_surov - (b + c * (x - x_sredina))

STARA METODA (točke modela):
    Polinom skozi točke surovega spektra, kjer je SME model >= PRAG_KONTINUUM
    (0.96), z iterativnim sigma clippingom. Normaliziran spekter: y_surov / C_old.

Za vsak spekter se nariše slika s 3 paneli:
    1. surovi spekter (brez kakršnihkoli vmesnih modifikacij) z obema kontinuumoma
    2. spekter, normaliziran z metodo (b, c), in SME model
    3. spekter, normaliziran s staro metodo, in SME model
V panelih 2 in 3 je v naslovu rms razlike (normaliziran spekter - SME model) na
pikslih, ki so bili uporabljeni v fitu (kovinske črte).

Uporaba:
    python normalizacija.py                 # vsi spektri iz tabele
    python normalizacija.py A2-342 A2-001   # samo izbrani objekti

Predpogoj: najprej zaženi sme_rv_vsi.py (tabela mora vsebovati b in c, določena
na surovih podatkih). Slike gredo v figs/kontinuum/.
"""

import glob
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")  # brez okna; mora biti pred uvozom pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

C_KMS = 299792.458  # hitrost svetlobe [km/s]

# ---------------------------------------------------------------
# NASTAVITVE (tukaj spreminjate parametre)
# ---------------------------------------------------------------
# Nova metoda: kako se uporabita b in c
NACIN_NORM_BC = "odstevanje"  # "deljenje": y / C_bc ; "odstevanje": y - (b + c*dx)

# Stara metoda (točke modela)
PRAG_KONTINUUM = 0.96  # vzame točke, kjer je SME model >= temu pragu
STOPNJA_POLINOMA = 1  # 1 = premica, 2 = parabola
SIGMA_CLIP = 2.0  # prag sigma clippinga
N_ITER_CLIP = 3  # število iteracij sigma clippinga

# Območja morajo biti ENAKA kot v sme_rv_vsi.py (da so x_sredina, fit in maska enaki)
OBMOCJE_OKNO = (5830.0, 5960.0)  # branje in prikaz (OBMOCJE_BRANJE)
FIT_OKNO = (5845.0, 5945.0)  # okno fita b, c (iz njega je x_sredina)
MASKA_ZVEZDNI = (5885.0, 5900.0)  # zvezdni Na D (rest valovne dolžine)
MASKA_ISM = (5888.5, 5897.5)  # mirujoči ISM Na D

# Poti do datotek
REZULTATI_TXT = "figs/sme_rv/rv_sme_rezultati.txt"
CACHE_FILE = "figs/sme_rv/modeli_cache.pkl"
SPEKTRI_GLOB = "Paper_II/*/*/*.fits"
IZHOD_SLIKE = "figs/kontinuum"

RESOLUCIJA = {"asiago": 24000, "varese": 11000}  # ločljivost R po instrumentu
RESOLUCIJA_DEFAULT = 20000


# ---------------------------------------------------------------
# BRANJE PODATKOV
# ---------------------------------------------------------------
def preberi_spekter(fits_path):
    """Prebere 1D spekter iz FITS datoteke in sestavi valovno skalo.

    Zakaj: spektri so shranjeni v različnih oblikah (1D ali 2D z enim redom);
    valovna skala se rekonstruira iz FITS glave (CRVAL1, CDELT1/CD1_1, CRPIX1).
    Enako kot v sme_rv_vsi.py.

    Args:
        fits_path: pot do FITS datoteke (str).

    Returns:
        (wave, flux): valovne dolžine [Å] in SUROV fluks (polja numpy, float).

    Raises:
        ValueError: če ima datoteka več redov (echelle) ali neznano obliko.
    """
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(float)
        h = hdul[0].header

    if data.ndim == 2:
        if data.shape[0] <= 4 and data.shape[1] > 1000:
            flux = data[0]
        else:
            raise ValueError(f"Echelle spekter z več redi {data.shape}")
    elif data.ndim == 1:
        flux = data
    else:
        flux = np.squeeze(data)
        if flux.ndim != 1:
            raise ValueError(f"Neznana dimenzija: {data.shape}")

    n_pix = len(flux)
    crval = h.get("CRVAL1", 0)
    cdelt = h.get("CDELT1", h.get("CD1_1", 1))
    crpix = h.get("CRPIX1", 1)
    wave = crval + (np.arange(n_pix) + 1 - crpix) * cdelt
    return wave, flux


def izrezi_okno(wave, flux):
    """Izreže območje OBMOCJE_OKNO iz surovega spektra (brez kakršnekoli obdelave).

    Zakaj: x_sredina mora biti povprečje valovnih dolžin v FIT_OKNO, enako kot
    pri fitu v sme_rv_vsi.py (tam je dx = x - povprečje x v oknu fita), sicer
    bi b in c opisovala drug kontinuum.

    Args:
        wave: valovne dolžine celotnega spektra [Å].
        flux: SUROV fluks celotnega spektra.

    Returns:
        (x, y, m_fit, x_sredina):
            x, y: valovne dolžine in surov fluks v OBMOCJE_OKNO,
            m_fit: logična maska pikslov, ki so v FIT_OKNO,
            x_sredina: povprečje x v FIT_OKNO [Å].

    Raises:
        ValueError: če spekter ne pokriva območja fita.
    """
    m = (
        (wave >= OBMOCJE_OKNO[0])
        & (wave <= OBMOCJE_OKNO[1])
        & np.isfinite(flux)
    )
    if (
        m.sum() < 50
        or wave.min() > FIT_OKNO[0] - 5
        or wave.max() < FIT_OKNO[1] + 5
    ):
        raise ValueError("spekter ne pokriva območja fita")

    x, y = wave[m], flux[m]
    m_fit = (x >= FIT_OKNO[0]) & (x <= FIT_OKNO[1])

    # --- VAROVALKA: Skaliranje le za absolutne flukse (_f z 1e-13) ---
    mediana_fluksa = np.nanmedian(y[m_fit])
    if 0 < mediana_fluksa < 0.01:
      y = y / mediana_fluksa

    x_sredina = np.mean(x[m_fit])
    return x, y, m_fit, x_sredina


# ---------------------------------------------------------------
# NOVA METODA (b, c)
# ---------------------------------------------------------------
def maska_brez_na(x, rv):
    """Uteži, ki izločijo zvezdni Na D (premaknjen za RV) in mirujoči ISM Na D.

    Zakaj: reproducira masko, s katero sta bila v sme_rv_vsi.py določena b in c
    (samo kovinske črte). Tu služi za prikaz maske in za izbiro pikslov pri rms.

    Args:
        x: valovne dolžine pikslov [Å].
        rv: RV zvezde [km/s]; zvezdna maska se premakne za ta RV.

    Returns:
        polje uteži: 0 znotraj zvezdne in ISM maske, sicer 1.
    """
    w = np.ones_like(x, dtype=float)
    faktor = 1.0 + rv / C_KMS
    zv_lo, zv_hi = MASKA_ZVEZDNI[0] * faktor, MASKA_ZVEZDNI[1] * faktor
    maska = ((x >= zv_lo) & (x <= zv_hi)) | ((x >= MASKA_ISM[0]) & (x <= MASKA_ISM[1]))
    w[maska] = 0.0
    return w


def kontinuum_bc(x, x_sredina, b, c):
    """Kontinuum nove metode: C_bc(x) = 1 + b + c * (x - x_sredina).

    Zakaj: model je na kontinuumu enak 1, zato je opazovani kontinuum 1 + b + c*dx
    (b in c iz fita surovega spektra).

    Args:
        x: valovne dolžine [Å].
        x_sredina: središče, glede na katerega je definiran naklon [Å].
        b: aditivni zamik iz fita.
        c: naklon iz fita [fluks/Å].

    Returns:
        kontinuum C_bc na x (v enotah surovega fluksa).
    """
    return 1.0 + b + c * (x - x_sredina)


def normaliziraj_bc(x, y, x_sredina, b, c):
    """Normalizira surov spekter z metodo (b, c).

    Zakaj: "deljenje" da y / C_bc (kontinuum = 1, črte pa so relativno plitvejše
    za faktor 1/(1+b)); "odstevanje" odšteje b + c*dx natanko tako, kot ga je
    odštel fit, zato je neposredno primerljiv z modelom. Izbira: NACIN_NORM_BC.

    Args:
        x: valovne dolžine [Å].
        y: surov fluks na x.
        x_sredina: središče za dx [Å].
        b: aditivni zamik iz fita.
        c: naklon iz fita [fluks/Å].

    Returns:
        (y_norm, C_bc): normaliziran spekter in kontinuum C_bc(x).
    """
    C_bc = kontinuum_bc(x, x_sredina, b, c)
    if NACIN_NORM_BC == "odstevanje":
        return y - (b + c * (x - x_sredina)), C_bc
    return y / C_bc, C_bc


# ---------------------------------------------------------------
# STARA METODA (točke modela)
# ---------------------------------------------------------------
def fit_kontinuum_polinom(x, y, stopnja=1, sigma=2.5, n_iter=3):
    """Iterativni fit polinoma z obojestranskim sigma clippingom.

    Zakaj: skozi točke kontinuuma želimo gladko krivuljo, ki je ne vlečejo
    osamelci (preostanki črt, šum). Po vsakem fitu se zavržejo točke, ki so od
    fita oddaljene za več kot sigma * std.

    Args:
        x: valovne dolžine točk [Å].
        y: fluks na teh točkah.
        stopnja: stopnja polinoma (1 = premica).
        sigma: prag clippinga v enotah standardnega odklona reziduala.
        n_iter: največ število iteracij.

    Returns:
        (coef, maska): koeficienti polinoma (np.polyfit) in logična maska točk,
        ki so ostale po clippingu.
    """
    maska_uporabljeno = np.ones_like(x, dtype=bool)

    for _ in range(n_iter):
        if maska_uporabljeno.sum() <= stopnja + 1:
            break
        coef = np.polyfit(x[maska_uporabljeno], y[maska_uporabljeno], stopnja)
        reziduali = y - np.polyval(coef, x)

        std = np.std(reziduali[maska_uporabljeno])
        if std == 0:
            break

        nova_maska = np.abs(reziduali) <= (sigma * std)
        if np.array_equal(maska_uporabljeno, nova_maska):
            break
        maska_uporabljeno = nova_maska

    coef_final = np.polyfit(x[maska_uporabljeno], y[maska_uporabljeno], stopnja)
    return coef_final, maska_uporabljeno


def kontinuum_stara_metoda(x, y, model_obs):
    """Kontinuum stare metode: polinom skozi točke, kjer je SME model >= prag.

    Zakaj: to je prvotni pristop (točke, kjer model nima črt, so na kontinuumu).
    Služi kot primerjava z metodo (b, c). Pomanjkljivost: v ozkem oknu je malo
    takih točk, na hladnih zvezdah (TiO) jih skoraj ni, polinom pa se zgreši.

    Args:
        x: valovne dolžine [Å].
        y: surov fluks na x.
        model_obs: SME model na x (že premaknjen za RV).

    Returns:
        (C_old, x_cont, y_cont, maska_clip):
            C_old: kontinuum (polinom) na x,
            x_cont, y_cont: točke, skozi katere je fitiran polinom,
            maska_clip: True za točke, ki so preživele sigma clipping.
    """
    maska = model_obs >= PRAG_KONTINUUM
    if maska.sum() < 10:  # prag previsok -> vzamemo zgornjih 5 % modela
        maska = model_obs >= np.percentile(model_obs, 95)

    x_cont, y_cont = x[maska], y[maska]
    coef, maska_clip = fit_kontinuum_polinom(
        x_cont, y_cont, stopnja=STOPNJA_POLINOMA, sigma=SIGMA_CLIP,
        n_iter=N_ITER_CLIP,
    )
    return np.polyval(coef, x), x_cont, y_cont, maska_clip


# ---------------------------------------------------------------
# PRIMERJAVA IN PRIKAZ
# ---------------------------------------------------------------
def rms_razlike(y_norm, model_obs, m_uporabljeno):
    """rms razlike med normaliziranim spektrom in SME modelom.

    Zakaj: ena številka za primerjavo metod normalizacije (manj = bolje se
    ujema z modelom).

    Args:
        y_norm: normaliziran spekter.
        model_obs: SME model na istih pikslih.
        m_uporabljeno: logična maska pikslov, vključenih v rms (kovinske črte
            v oknu fita, brez maskiranega Na D).

    Returns:
        rms (float).
    """
    return float(np.sqrt(np.mean((y_norm[m_uporabljeno] - model_obs[m_uporabljeno]) ** 2)))


def narisi_primerjavo(x, y, C_bc, C_old, y_bc, y_old, model_obs, w,
                      vrstica, b, c, x_cont, y_cont, maska_clip,
                      rms_bc, rms_old, izhod_pot):
    """Nariše sliko s 3 paneli, ki primerja novo (b, c) in staro metodo.

    Zakaj: nazorna primerjava za zagovor. Zgoraj surov spekter z obema
    kontinuumoma; v sredini in spodaj normaliziran spekter po vsaki metodi z
    istim SME modelom in istim y razponom.

    Args:
        x: valovne dolžine [Å].
        y: surov fluks.
        C_bc: kontinuum nove metode na x.
        C_old: kontinuum stare metode na x.
        y_bc: spekter, normaliziran z metodo (b, c).
        y_old: spekter, normaliziran s staro metodo.
        model_obs: SME model na x (premaknjen za RV).
        w: uteži maske Na D (0 = maskirano); za prikaz oranžnega območja.
        vrstica: vrstica tabele rezultatov (objekt, instrument, spekter,
            rv_sme, teff, monh, dna).
        b: aditivni zamik iz fita.
        c: naklon iz fita [fluks/Å].
        x_cont, y_cont: točke, uporabljene za kontinuum stare metode.
        maska_clip: True za točke, ki so preživele sigma clipping.
        rms_bc: rms (spekter - model) za metodo (b, c).
        rms_old: rms (spekter - model) za staro metodo.
        izhod_pot: pot izhodne PNG slike.
    """
    objekt, instrument = vrstica["objekt"], vrstica["instrument"]
    rv = vrstica["rv_sme"]
    ime_modela = (
        f"SME model (Teff={vrstica['teff']:.0f}, "
        f"[M/H]={vrstica['monh']:+.1f}, dNa={vrstica['dna']:+.1f})"
    )
    oznaka_bc = (
        "y - (b + c*dx)" if NACIN_NORM_BC == "odstevanje" else "y / C(b, c)"
    )

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 11.5), sharex=True,
        gridspec_kw=dict(height_ratios=[1.0, 1.0, 1.0]),
    )

    m_mask = w == 0.0
    for ax in (ax1, ax2, ax3):
        if m_mask.any():
            ax.axvspan(x[m_mask].min(), x[m_mask].max(), color="orange", alpha=0.13)
        ax.axvline(FIT_OKNO[0], color="gray", ls=":", lw=0.8)
        ax.axvline(FIT_OKNO[1], color="gray", ls=":", lw=0.8)
        ax.grid(alpha=0.3)

    # --- 1. Surovi spekter + oba kontinuuma ---
    ax1.plot(x, y, color="gray", lw=1.0, label="Surovi spekter (FITS)")
    ax1.plot(x, C_bc, color="darkred", lw=2.2,
             label=f"Kontinuum (b, c): 1 + b + c*dx;  b={b:+.3f}, c={c * 100:+.4f}/100A")
    ax1.plot(x, C_old, color="royalblue", lw=1.8, ls="--",
             label=f"Kontinuum (stara metoda): polinom st. {STOPNJA_POLINOMA} skozi SME >= {PRAG_KONTINUUM}")
    if len(x_cont) > 0:
        ax1.scatter(x_cont[maska_clip], y_cont[maska_clip], color="forestgreen",
                    s=14, zorder=4, label=f"Točke SME >= {PRAG_KONTINUUM} (N={maska_clip.sum()})")
        if (~maska_clip).sum() > 0:
            ax1.scatter(x_cont[~maska_clip], y_cont[~maska_clip], color="red",
                        marker="x", s=24, zorder=5,
                        label=f"Sigma clipped (N={(~maska_clip).sum()})")
    ax1.set_ylabel("Fluks (surov)")
    ax1.set_ylim(min(np.nanmin(y), 0.0) - 0.05,
                 max(np.nanmax(y), np.nanmax(C_bc), np.nanmax(C_old)) + 0.1)
    ax1.set_title(
        f"{objekt} ({instrument}) - {vrstica['spekter']} | RV = {rv:.2f} km/s   "
        f"(oranžno: maskiran Na D; pikčasto: okno fita)",
        fontsize=11,
    )
    ax1.legend(loc="lower left", fontsize=8, framealpha=0.92)

    # --- 2. Normaliziran z (b, c) ---
    ax2.plot(x, y_bc, color="black", lw=1.0, label=f"Normaliziran: {oznaka_bc}")
    ax2.plot(x, model_obs, color="crimson", lw=1.1, ls="--", alpha=0.9, label=ime_modela)
    ax2.axhline(1.0, color="gray", ls=":", lw=1.2)
    ax2.set_ylabel("Normaliziran fluks")
    ax2.set_title(f"Nova metoda (b, c): rms(spekter - model) = {rms_bc:.4f}", fontsize=11)
    ax2.legend(loc="lower left", fontsize=8, framealpha=0.92)

    # --- 3. Normaliziran s staro metodo ---
    ax3.plot(x, y_old, color="black", lw=1.0,
             label=f"Normaliziran: y / C(SME >= {PRAG_KONTINUUM})")
    ax3.plot(x, model_obs, color="crimson", lw=1.1, ls="--", alpha=0.9, label=ime_modela)
    ax3.axhline(1.0, color="gray", ls=":", lw=1.2)
    ax3.set_xlabel(r"Valovna dolžina [$\AA$]")
    ax3.set_ylabel("Normaliziran fluks")
    ax3.set_title(f"Stara metoda (točke modela, prag {PRAG_KONTINUUM}, sigma clipping): "
                  f"rms(spekter - model) = {rms_old:.4f}", fontsize=11)
    ax3.legend(loc="lower left", fontsize=8, framealpha=0.92)

    # Enak y razpon obeh normaliziranih panelov za neposredno primerjavo
    y_lo = min(0.0, np.nanmin(y_bc), np.nanmin(y_old)) - 0.05
    y_hi = max(1.25, np.nanmax(y_bc) + 0.05, np.nanmax(y_old) + 0.05)
    for ax in (ax2, ax3):
        ax.set_ylim(y_lo, y_hi)
    ax3.set_xlim(*OBMOCJE_OKNO)

    plt.tight_layout()
    plt.savefig(izhod_pot, dpi=100)
    plt.close()


# ---------------------------------------------------------------
# GLAVNI PROGRAM
# ---------------------------------------------------------------
def main(filter_objekti=None):
    """Za vse (ali izbrane) spektre iz tabele izračuna obe normalizaciji in nariše slike.

    Zakaj: vstopna točka skripte. Za vsak spekter prebere surov FITS, vzame RV,
    model, b in c iz tabele rv_sme_rezultati.txt (sme_rv_vsi.py), izračuna
    kontinuum po obeh metodah in shrani primerjalno sliko.

    Args:
        filter_objekti: seznam imen objektov (velike črke, npr. ["A2-342"]);
            če None, se obdelajo vsi objekti iz tabele.
    """
    os.makedirs(IZHOD_SLIKE, exist_ok=True)

    if not os.path.exists(CACHE_FILE):
        print(f"Napaka: Datoteka {CACHE_FILE} ne obstaja!")
        return
    if not os.path.exists(REZULTATI_TXT):
        print(f"Napaka: {REZULTATI_TXT} ne obstaja. Najprej zaženite sme_rv_vsi.py!")
        return

    print("1. Nalagam shranjene SME modele...")
    with open(CACHE_FILE, "rb") as f:
        modeli_cache = pickle.load(f)

    print("2. Nalagam rezultate RV meritev...")
    df = pd.read_csv(REZULTATI_TXT, sep=r"\s+", comment="#")
    if filter_objekti:
        df = df[df["objekt"].str.upper().isin(filter_objekti)]
    print(f"   Spektrov za obdelavo: {len(df)}")

    # Indeks FITS datotek (objekt, instrument, ime) -> pot; enako kot sme_rv_vsi.py
    poti = {}
    for p in glob.glob(SPEKTRI_GLOB):
        deli = p.replace("\\", "/").split("/")
        poti[(deli[-3].upper(), deli[-2].lower(), deli[-1])] = p

    uspesno = preskoceno = 0
    for _, vrstica in df.iterrows():
        objekt = str(vrstica["objekt"]).upper()
        instrument = str(vrstica["instrument"]).lower()
        ime = str(vrstica["spekter"])

        fits_pot = poti.get((objekt, instrument, ime))
        if fits_pot is None:
            print(f"   [Opozorilo] {ime} ni na disku. Preskakujem.")
            preskoceno += 1
            continue

        try:
            wave, flux = preberi_spekter(fits_pot)
            x, y, m_fit, x_sredina = izrezi_okno(wave, flux)
        except Exception as e:
            print(f"   [Napaka] {ime}: {e}")
            preskoceno += 1
            continue

        res = RESOLUCIJA.get(instrument, RESOLUCIJA_DEFAULT)
        key = (res, float(vrstica["teff"]), float(vrstica["monh"]),
               float(vrstica["dna"]))
        if key not in modeli_cache:
            print(f"   [Opozorilo] Model {key} ni v predpomnilniku ({ime}). Preskakujem.")
            preskoceno += 1
            continue
        wr, syn = modeli_cache[key]

        # SME model premaknjen za izmerjeni RV (isto kot v chi2_pri_rv)
        rv = float(vrstica["rv_sme"])
        model_obs = np.interp(x / (1.0 + rv / C_KMS), wr, syn)

        # NOVA METODA: b in c iz tabele (določena na surovem fluksu)
        b, c = float(vrstica["b_offset"]), float(vrstica["c_slope"])
        y_bc, C_bc = normaliziraj_bc(x, y, x_sredina, b, c)

        # STARA METODA: polinom skozi točke SME >= prag + sigma clipping
        C_old, x_cont, y_cont, maska_clip = kontinuum_stara_metoda(x, y, model_obs)

        if np.any(C_bc <= 0) or np.any(C_old <= 0):
            print(f"   [Opozorilo] {ime}: kontinuum <= 0, preskakujem.")
            preskoceno += 1
            continue
        y_old = y / C_old

        # rms samo na pikslih, ki so bili uporabljeni v fitu (okno fita, brez Na D)
        w = maska_brez_na(x, rv)
        m_uporabljeno = (w * m_fit) > 0
        rms_bc = rms_razlike(y_bc, model_obs, m_uporabljeno)
        rms_old = rms_razlike(y_old, model_obs, m_uporabljeno)

        izhod = os.path.join(
            IZHOD_SLIKE, f"{objekt}_{instrument}_{ime.replace('.fits', '')}_kontinuum.png"
        )
        narisi_primerjavo(x, y, C_bc, C_old, y_bc, y_old, model_obs, w,
                          vrstica, b, c, x_cont, y_cont, maska_clip,
                          rms_bc, rms_old, izhod)

        uspesno += 1
        if uspesno % 25 == 0:
            print(f"   Obdelanih {uspesno}/{len(df)} spektrov...")

    print(f"\nKončano! Uspešno: {uspesno}, preskočenih: {preskoceno}.")
    print(f"Slike so v: {IZHOD_SLIKE}/")


if __name__ == "__main__":
    main([a.upper() for a in sys.argv[1:]] or None)