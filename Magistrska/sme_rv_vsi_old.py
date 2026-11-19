"""
SME fit RV za vse spektre - parametri modela prilagojeni VSAKEMU OBJEKTU.

Postopek:
  1. Mreža modelov (teff x monh x dNa) se sintetizira ENKRAT (skupna za vse
     objekte) in se shrani na disk (modeli_cache.pkl) - ponovni zagon je hiter.
  2. Za vsak objekt:
       a) izberemo do N_IZBOR spektrov (enakomerno po času),
       b) za vsak model iz mreže poiščemo najboljši RV (+-5 km/s okoli lit.)
          in seštejemo log(chi2_min) čez izbrane spektre,
       c) model z najmanjšo vsoto = model tega objekta.
  3. Ta model se uporabi za RV fit VSEH spektrov tega objekta.

Izhod:
  rv_sme_rezultati.txt      - tabela po spektrih + povzetek po objektih
  rv_sme_modeli_objektov.txt- izbrani parametri modela za vsak objekt
  figs/sme_rv/             - slika fita za vsak spekter
"""
import os
import glob
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from pysme.sme import SME_Structure
from pysme.abund import Abund
from pysme.linelist.vald import ValdFile
from pysme.synthesize import synthesize_spectrum

C_KMS = 299792.458
NA_D2, NA_D1 = 5889.950, 5895.924

# ---------------------------------------------------------------
# NASTAVITVE
# ---------------------------------------------------------------
VALD_PATH = "vald_3800K_5800_6000"
RV_CSV = "zvezde_RV_iz_literature.csv"
SPEKTRI_GLOB = "Paper_II/*/*/*.fits"
IZHOD_TXT = "figs/sme_rv/rv_sme_rezultati.txt"
MODELI_TXT = "figs/sme_rv/rv_sme_modeli_objektov.txt"
CACHE_FILE = "figs/sme_rv/modeli_cache.pkl"
SLIKE_DIR = "figs/sme_rv"
RISI_SLIKE = True


OBMOCJE_SINTEZE = (5820.0, 5970.0)  # SME model (za normalizacijo kontinuuma pozneje)
OBMOCJE_BRANJE = (5830.0, 5960.0)   # za detrend kontinuuma
FIT_OKNO = (5878.0, 5908.0)         # tu se fitta RV (obdrži ozko na Na D!)
RV_OKNO =35.0                        # +- km/s okoli literaturne vrednosti
RV_KORAK = 0.05                     # korak za končni fit
RV_KORAK_IZBOR = 0.2                # grobi korak pri izbiri modela
RV_OKNO_WIDE = 100.0                # +- km/s, drugi prehod za spektre z na_robu=True
RV_KORAK_WIDE = 0.5                 # grobi korak v širokem prehodu
ODSTOP_PRAG = 12.0                  # km/s od mediane objekta -> zastavica 'odstopa'
#MASKA_NA_JEDRA = 1.0               # +- Å okoli tmirovnih Na D (ISM)
MARGIN_MODEL = 20.0                 # Å rezerve modela (RV do ~ +-600 km/s)
DLAM_MODEL = 0.01                   # Å, korak mreže modela
N_IZBOR = 7                         # št. spektrov za izbiro modela objekta

# Fiksni parametri modela
LOGG, VMIC, VMAC, VSINI = 1.0, 2.0, 3.0, 10.0

# Resolucija po inštrumentu (ime mape, male črke)
RESOLUCIJA = {"asiago": 24000, "varese": 11000}
RESOLUCIJA_DEFAULT = 20000

# MREŽA prostih parametrov (skupna vsem objektom; 7*6*6 = 252 modelov na R) 
TEFF_MREZA = [2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600.0, 3700.0, 3800.0, 3900.0]
MONH_MREZA = [-0.8, -0.4, 0.0, 0.2, 0.4, 0.6]
NA_MREZA = [-0.3, 0.0, 0.3, 0.6, 0.9, 1.2]       # odmik [Na/H] glede na monh


# ---------------------------------------------------------------
# POMOŽNE FUNKCIJE
# ---------------------------------------------------------------
def preberi_spekter(fits_path):
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(float)
        h = hdul[0].header

    # Če ima spekter več dimenzij (npr. (1, N) ali (2, N))
    if data.ndim == 2:
        # Primer Asiago (2, N): prva vrstica je spekter, druga je napaka/nebo
        if data.shape[0] <= 4 and data.shape[1] > 1000:
            flux = data[0]
        # Primer Varese (27, 740): gre za nezdržen echelle spekter
        else:
            raise ValueError(f"Echelle spekter z več redi {data.shape} - potrebuje 1D združitev (merge)")
    elif data.ndim == 1:
        flux = data
    else:
        # Če je npr. oblika (1, 1, N)
        flux = np.squeeze(data)
        if flux.ndim != 1:
            raise ValueError(f"Neznana oblika podatkov: {data.shape}")

    # Dolžino vzamemo iz dejanskega 1D fluksa
    n_pix = len(flux)
    crval = h.get("CRVAL1", 0)
    cdelt = h.get("CDELT1", h.get("CD1_1", 1))
    crpix = h.get("CRPIX1", 1)
    wave = crval + (np.arange(n_pix) + 1 - crpix) * cdelt

    return wave, flux


def nalozi_literaturne_rv(csv_path):
    df = pd.read_csv(csv_path)
    slovar = {}
    for _, v in df.iterrows():
        if pd.notna(v.get("RV_GALAH [km/s]")):
            slovar[v["ID"].upper()] = (float(v["RV_GALAH [km/s]"]), "GALAH")
        elif pd.notna(v.get("RV_Gaia [km/s]")):
            slovar[v["ID"].upper()] = (float(v["RV_Gaia [km/s]"]), "Gaia")
    return slovar


def sintetiziraj(wave_rest, linelist, teff, monh, dna, res):
    sme = SME_Structure()
    sme.wave = wave_rest
    sme.spec = np.ones_like(wave_rest)
    sme.uncs = np.ones_like(wave_rest) * 0.01
    sme.teff, sme.logg, sme.vmic = teff, LOGG, VMIC
    sme.vmac, sme.vsini = VMAC, VSINI
    sme.monh = monh
    sme.abund = Abund(monh, "asplund2009")
    sme.abund["Na"] = sme.abund["Na"] + dna
    sme.linelist = linelist
    sme.iptype = "gauss"
    sme.ipres = res
    sme.vrad = 0.0
    sme.vrad_flag = "none"
    sme.cscale_flag = "none"
    sme = synthesize_spectrum(sme)
    return np.asarray(sme.wave[0]), np.asarray(sme.synth[0])


def chi2_pri_rv(v, x, y, w, wave_rest, synth):
    model = np.interp(x / (1.0 + v / C_KMS), wave_rest, synth)
    dx = x - x.mean()
    A = np.vstack([model, model * dx]).T
    coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    fit = A @ coef
    return np.sum(((y - fit) * w) ** 2), fit


class ModelCache:
    """Model (res, teff, monh, dNa) se sintetizira samo enkrat; shrani se na disk."""
    def __init__(self, linelist, cache_file):
        self.linelist = linelist
        self.cache_file = cache_file

        self.wave_rest = np.arange(OBMOCJE_SINTEZE[0], OBMOCJE_SINTEZE[1], DLAM_MODEL)

        self.cache = {}
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.cache = pickle.load(f)
            print(f"Naloženih {len(self.cache)} modelov iz {cache_file}")
        self.nov = 0

    def get(self, res, teff, monh, dna):
        key = (res, teff, monh, dna)
        if key not in self.cache:
            print(f"    [SME sinteza] R={res}, Teff={teff:.0f}, monh={monh:+.1f}, dNa={dna:+.1f}")
            self.cache[key] = sintetiziraj(self.wave_rest, self.linelist,
                                           teff, monh, dna, res)
            self.nov += 1
            if self.nov % 10 == 0:
                self.shrani()
        return self.cache[key]

    def shrani(self):
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache, f)


class Spekter:
    """Pripravljen spekter (detrendan, v oknu fita, z masko)."""
    def __init__(self, fits_path, objekt, instrument):
        self.path = fits_path
        self.ime = os.path.basename(fits_path)
        self.objekt, self.instrument = objekt, instrument
        wave, flux = preberi_spekter(fits_path)
        m = (wave > OBMOCJE_BRANJE[0]) & (wave < OBMOCJE_BRANJE[1]) & np.isfinite(flux)
        if m.sum() < 50 or wave.min() > FIT_OKNO[0] - 5 or wave.max() < FIT_OKNO[1] + 5:
            raise ValueError("spekter ne pokriva območja Na D")
        x, y = wave[m], flux[m]
        m_cont = ((x > 5830) & (x < 5880)) | ((x > 5905) & (x < 5960))
        y = y / np.polyval(np.polyfit(x[m_cont], y[m_cont], 1), x)
        m_fit = (x > FIT_OKNO[0]) & (x < FIT_OKNO[1])
        self.x, self.y = x[m_fit], y[m_fit]
        # ODSTRANIMO MASKO ZA SME!
        #self.maska = np.ones_like(self.x, dtype=bool)
        #for lam in (NA_D2, NA_D1):
        #    self.maska &= np.abs(self.x - lam) > MASKA_NA_JEDRA
        #self.w = self.maska.astype(float)
        # Vse točke imajo polno utež (brez maskiranja ISM jeder)
        self.maska = np.ones_like(self.x, dtype=bool)
        self.w = np.ones_like(self.x, dtype=float)
        self.res = RESOLUCIJA.get(instrument, RESOLUCIJA_DEFAULT)


def skeniraj_rv(sp, model, rv_lit, korak, okno=None):
    wr, syn = model
    okno = RV_OKNO if okno is None else okno
    rv_grid = np.arange(rv_lit - okno, rv_lit + okno + korak, korak)
    chi2 = np.array([chi2_pri_rv(v, sp.x, sp.y, sp.w, wr, syn)[0] for v in rv_grid])
    return rv_grid, chi2


def parabola_min(rv_grid, chi2):
    """Minimum chi2 s parabolo okoli najnižje točke. Vrne (rv, a, i_min)."""
    i_min = int(np.argmin(chi2))
    sl = slice(max(i_min - 10, 0), min(i_min + 11, len(rv_grid)))
    a, b, c = np.polyfit(rv_grid[sl], chi2[sl], 2)
    rv = -b / (2 * a) if a > 0 else rv_grid[i_min]
    return rv, a, i_min


def siroki_prehod(sp, model, rv_lit):
    """Drugi prehod: grobo skeniranje +-RV_OKNO_WIDE, nato fino okoli minimuma.
    Vrne (rv, napaka, na_robu_wide)."""
    wr, syn = model
    g, c = skeniraj_rv(sp, model, rv_lit, RV_KORAK_WIDE, RV_OKNO_WIDE)
    rv0 = g[int(np.argmin(c))]
    na_robu_w = int(np.argmin(c)) in (0, len(g) - 1)
    g2, c2 = skeniraj_rv(sp, model, rv0, RV_KORAK, 3.0)
    rv, a, _ = parabola_min(g2, c2)
    chi2_min, _ = chi2_pri_rv(rv, sp.x, sp.y, sp.w, wr, syn)
    sigma = np.sqrt(chi2_min / (int(sp.w.sum()) - 3))
    e = sigma / np.sqrt(a) if a > 0 else np.nan
    return rv, e, na_robu_w


def izberi_model_objekta(spektri, rv_lit, modeli):
    """Poišče (teff, monh, dNa), ki najbolje opiše izbrane spektre objekta."""
    if len(spektri) > N_IZBOR:
        idx = np.linspace(0, len(spektri) - 1, N_IZBOR).astype(int)
        izbor = [spektri[i] for i in idx]
    else:
        izbor = spektri

    rezultati = {}
    for teff, monh, dna in itertools.product(TEFF_MREZA, MONH_MREZA, NA_MREZA):
        skupaj = 0.0
        for sp in izbor:
            model = modeli.get(sp.res, teff, monh, dna)
            _, chi2 = skeniraj_rv(sp, model, rv_lit, RV_KORAK_IZBOR)
            skupaj += np.log(chi2.min())
        rezultati[(teff, monh, dna)] = skupaj / len(izbor)
    best = min(rezultati, key=rezultati.get)
    return best, rezultati[best]


def fit_spekter(sp, rv_lit, vir, par, modeli):
  teff, monh, dna = par
  wr, syn = modeli.get(sp.res, teff, monh, dna)

  # 1. Prvi poskus z običajnim oknom
  rv_grid, chi2 = skeniraj_rv(sp, (wr, syn), rv_lit, RV_KORAK)
  i_min = int(np.argmin(chi2))
  sl = slice(max(i_min - 10, 0), min(i_min + 11, len(rv_grid)))
  a, b, c = np.polyfit(rv_grid[sl], chi2[sl], 2)
  rv_sme = -b / (2 * a) if a > 0 else rv_grid[i_min]

  chi2_min, fit = chi2_pri_rv(rv_sme, sp.x, sp.y, sp.w, wr, syn)
  sigma = np.sqrt(chi2_min / (int(sp.w.sum()) - 3))
  e_rv = sigma / np.sqrt(a) if a > 0 else np.nan
  na_robu = i_min in (0, len(rv_grid) - 1)

  # -------------------------------------------------------------
  # POPRAVEK: Če je minimum na robu, avtomatsko prevzamemo široki fit!
  # -------------------------------------------------------------
  rv_wide, e_wide, wide_robu = np.nan, np.nan, False
  if na_robu:
    rv_wide, e_wide, wide_robu = siroki_prehod(sp, (wr, syn), rv_lit)

    # Če je široki fit našel pravi minimum (ni obstal na robu 100 km/s):
    if not wide_robu and np.isfinite(rv_wide):
      rv_sme = rv_wide
      e_rv = e_wide
      na_robu = False  # uspešno rešeno!

      # Ponovno izračunamo fit z novim RV za sliko
      chi2_min, fit = chi2_pri_rv(rv_sme, sp.x, sp.y, sp.w, wr, syn)

      # Ponovno ustvarimo lepo centrirano mrežo chi2 okoli novega RV za spodnji graf
      rv_grid, chi2 = skeniraj_rv(
          sp, (wr, syn), rv_sme, RV_KORAK, okno=RV_OKNO
      )

  rez = dict(
      objekt=sp.objekt,
      instrument=sp.instrument,
      spekter=sp.ime,
      rv_lit=rv_lit,
      vir=vir,
      rv_sme=rv_sme,
      e_rv_sme=e_rv,
      razlika=rv_sme - rv_lit,
      teff=teff,
      monh=monh,
      dna=dna,
      sigma=sigma,
      na_robu=na_robu,
      rv_sme_wide=rv_wide,
      e_rv_wide=e_wide,
      wide_na_robu=wide_robu,
  )

  if RISI_SLIKE:
    narisi(sp, fit, rv_grid, chi2, rv_lit, vir, rv_sme, e_rv, rez)

  return rez

def narisi(sp, fit, rv_grid, chi2, rv_lit, vir, rv_sme, e_rv, rez):
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw=dict(height_ratios=[2, 1]))
    ax[0].plot(sp.x, sp.y, "k-", lw=1.2, label="Opazovano")
    ax[0].plot(sp.x, fit, "r--", lw=1.8,
               label=(f"SME (RV={rv_sme:.2f}, Teff={rez['teff']:.0f}, "
                      f"monh={rez['monh']:+.1f}, dNa={rez['dna']:+.1f})"))
    #ax[0].plot(sp.x[~sp.maska], sp.y[~sp.maska], ".", color="orange", ms=3, label="Maskirano (ISM)")
    ax[0].set_xlim(*FIT_OKNO)
    ax[0].set_title(f"{rez['objekt']} – {rez['spekter']}")
    ax[0].legend(loc="lower left")
    ax[0].grid(alpha=0.3)
    ax[1].plot(rv_grid, chi2, "b-")
    ax[1].axvline(rv_lit, color="g", ls="--", label=f"{vir}: {rv_lit:.2f}")
    ax[1].axvline(rv_sme, color="r", ls="--", label=f"SME: {rv_sme:.2f} ± {e_rv:.2f}")
    ax[1].set_xlabel("RV (km/s)")
    ax[1].set_ylabel(r"$\chi^2$")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    ime = rez["spekter"].replace(".fits", "")
    plt.savefig(os.path.join(SLIKE_DIR, f"{rez['objekt']}_{rez['instrument']}_{ime}.png"), dpi=90)
    plt.close()


def zapisi_txt(rezultati, path):
    df = pd.DataFrame(rezultati)
    # Zastavica 'odstopa': |RV_SME - mediana veljavnih fitov objekta| > ODSTOP_PRAG.
    # Za spektre z na_robu se uporabi rv_sme_wide (če obstaja).
    med = df[~df["na_robu"]].groupby("objekt")["rv_sme"].median()
    rv_ref = df["rv_sme"].where(~df["na_robu"], df["rv_sme_wide"])
    df["odstopa"] = [
        bool(o in med.index and np.isfinite(r) and abs(r - med[o]) > ODSTOP_PRAG)
        for o, r in zip(df["objekt"], rv_ref)
    ]
    with open(path, "w") as f:
        f.write("# RV iz SME (brez solve, model prilagojen vsakemu objektu) vs literatura\n")
        f.write("# razlika = RV_SME - RV_lit; na_robu=True -> minimum na robu okna, fit NI vreden\n")
        f.write("# rv_sme_wide = drugi prehod (+-%.0f km/s), samo za na_robu=True; "
                "wide_na_robu=True -> tudi tam na robu\n" % RV_OKNO_WIDE)
        f.write("# odstopa=True -> RV je več kot %.0f km/s od mediane veljavnih fitov objekta\n"
                % ODSTOP_PRAG)
        f.write(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        f.write("\n")
        ok = df[~df["na_robu"] & ~df["odstopa"]]
        if len(ok):
            f.write(f"\n# Veljavni fiti (brez roba in brez odstopanj): {len(ok)}/{len(df)}\n")
            f.write(f"# Razlika SME-lit: mean={ok['razlika'].mean():.3f}, "
                    f"median={ok['razlika'].median():.3f}, std={ok['razlika'].std():.3f} km/s\n")
            for ime, g in ok.groupby("objekt"):
                f.write(f"# {ime}: N={len(g)}  mean_diff={g['razlika'].mean():+.2f}  "
                        f"std_RV_SME={g['rv_sme'].std():.2f}\n")
        sumljivi = df[df["na_robu"] | df["odstopa"]]
        if len(sumljivi):
            f.write(f"\n# SUMLJIVI spektri ({len(sumljivi)}):\n")
            for _, r in sumljivi.iterrows():
                f.write(f"# {r['objekt']} {r['spekter']}: rv_sme={r['rv_sme']:.2f} "
                        f"wide={r['rv_sme_wide']:.2f} lit={r['rv_lit']:.2f} "
                        f"na_robu={r['na_robu']} odstopa={r['odstopa']}\n")


def zapisi_modele(modeli_obj, path):
    robovi = dict(teff=(min(TEFF_MREZA), max(TEFF_MREZA)),
                  monh=(min(MONH_MREZA), max(MONH_MREZA)),
                  dna=(min(NA_MREZA), max(NA_MREZA)))
    with open(path, "w") as f:
        f.write("# Izbrani parametri modela po objektih (* = na robu mreže)\n")
        f.write(f"{'objekt':8s} {'Teff':>6s} {'monh':>6s} {'dNa':>6s} {'N_spektrov':>10s}\n")
        for obj, (par, n) in sorted(modeli_obj.items()):
            teff, monh, dna = par
            zvezd = lambda v, k: "*" if v in robovi[k] else " "
            f.write(f"{obj:8s} {teff:6.0f}{zvezd(teff,'teff')} {monh:+6.1f}{zvezd(monh,'monh')} "
                    f"{dna:+6.1f}{zvezd(dna,'dna')} {n:10d}\n")


# ---------------------------------------------------------------
# GLAVNI DEL
# ---------------------------------------------------------------
def main():
    if RISI_SLIKE:
        os.makedirs(SLIKE_DIR, exist_ok=True)
    rv_slovar = nalozi_literaturne_rv(RV_CSV)
    modeli = ModelCache(ValdFile(VALD_PATH), CACHE_FILE)

    # Razvrsti spektre po objektih
    poti = sorted(glob.glob(SPEKTRI_GLOB))
    po_objektih = {}
    preskoceni = []
    for p in poti:
        deli = p.replace("\\", "/").split("/")
        objekt, instrument = deli[-3].upper(), deli[-2].lower()
        if objekt not in rv_slovar:
            preskoceni.append((p, "ni literaturnega RV"))
            continue
        try:
            po_objektih.setdefault(objekt, []).append(Spekter(p, objekt, instrument))
        except Exception as e:
            preskoceni.append((p, str(e)))
    print(f"Objektov: {len(po_objektih)}, spektrov: {sum(len(v) for v in po_objektih.values())}")

    rezultati, modeli_obj = [], {}
    for k, (objekt, spektri) in enumerate(sorted(po_objektih.items()), 1):
        rv_lit, vir = rv_slovar[objekt]
        print(f"\n[{k}/{len(po_objektih)}] {objekt}: {len(spektri)} spektrov, RV_lit={rv_lit:.2f} ({vir})")
        par, score = izberi_model_objekta(spektri, rv_lit, modeli)
        modeli_obj[objekt] = (par, len(spektri))
        print(f"    Izbran model: Teff={par[0]:.0f}, monh={par[1]:+.1f}, dNa={par[2]:+.1f}")

        for sp in spektri:
            try:
                r = fit_spekter(sp, rv_lit, vir, par, modeli)
                rezultati.append(r)
                opoz = "  !! NA ROBU" if r["na_robu"] else ""
                print(f"    {sp.ime}: RV_SME={r['rv_sme']:.2f}±{r['e_rv_sme']:.2f} "
                      f"diff={r['razlika']:+.2f}{opoz}")
            except Exception as e:
                preskoceni.append((sp.path, str(e)))
                print(f"    {sp.ime}: PRESKOČEN: {e}")

        modeli.shrani()
        zapisi_txt(rezultati, IZHOD_TXT)
        zapisi_modele(modeli_obj, MODELI_TXT)

    print(f"\nKončano. Rezultati: {IZHOD_TXT}, modeli: {MODELI_TXT}")
    if preskoceni:
        with open("rv_sme_preskoceni.txt", "w") as f:
            for p, razlog in preskoceni:
                f.write(f"{p}\t{razlog}\n")
        print(f"Preskočenih: {len(preskoceni)} (rv_sme_preskoceni.txt)")


if __name__ == "__main__":
    main()
