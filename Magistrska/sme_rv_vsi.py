"""
SME fit radialne hitrosti (RV) za vse spektre z aditivnim offsetom in naklonom.

Postopek:
  1. Spektri (_c.fits) so že pred-obdelani na telesku (fluks okoli 1), zato se v
     fit vstavi SUROV fluks iz FITS datoteke, brez kakršnekoli normalizacije.
  2. Za vsak objekt se iz mreže (Teff x [M/H] x dNa) izbere en sintetični SME
     model (izberi_model_objekta).
  3. RV se za vsak spekter določi v dveh stopnjah:
       - 1. stopnja: grobi RV (+-RV_OKNO_GROBO km/s okoli literaturnega RV), z Na D.
       - 2. stopnja: fini RV okoli grobega, z maskiranim Na D (samo kovinske črte).
  4. V vsakem chi2 se model fitira na surov fluks:
         y_surov ~= model + b + c * (x - x_sredina)
     (aditivni zamik b in naklon c, brez množenja modela). Končna b_offset in
     c_slope sta shranjena v tabelo in se uporabita v normalizacija.py:
         C(x) = 1 + b + c * (x - x_sredina)

Izhodi (v figs/sme_rv/): rv_sme_rezultati.txt, rv_sme_modeli_objektov.txt,
modeli_cache.pkl (predpomnjeni SME modeli - ne brisati, sinteza je draga) in slike
fitov za vsak spekter.
"""

import glob
import itertools
import os
import pickle

import matplotlib

matplotlib.use("Agg")  # brez okna; mora biti pred uvozom pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from pysme.abund import Abund
from pysme.linelist.vald import ValdFile
from pysme.sme import SME_Structure
from pysme.synthesize import synthesize_spectrum

C_KMS = 299792.458  # hitrost svetlobe [km/s]

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

# Valovna območja [Å]
OBMOCJE_SINTEZE = (5820.0, 5970.0)  # sintetični model (rezerva za premik z RV)
OBMOCJE_BRANJE = (5830.0, 5960.0)  # kar preberemo iz FITS
FIT_OKNO = (5845.0, 5945.0)  # okno fita (Na D + okoliške kovinske črte)
DLAM_MODEL = 0.01  # korak valovne mreže modela [Å]

# Iskanje RV
RV_OKNO_GROBO = 40.0  # +- km/s za 1. stopnjo (ujame faze dvozvezdij)
RV_KORAK_GROBO = 0.5  # korak 1. stopnje [km/s]
RV_OKNO_FINO = 6.0  # +- km/s okoli grobega RV za 2. stopnjo
RV_KORAK_FINO = 0.05  # korak 2. stopnje [km/s] (za fit parabole)
N_IZBOR = 5  # št. spektrov, na katerih se izbere model objekta

# Izbira modela: cenejše iskanje RV (isti postopek kot končni fit)
IZB_KORAK_GROBO = 1.0  # korak grobe stopnje [km/s]
IZB_OKNO_FINO = 3.0  # +- okno fine stopnje [km/s]
IZB_KORAK_FINO = 0.25  # korak fine stopnje [km/s]

# Kazen za offset b pri izbiri modela (b v končnem fitu ostane prost):
# do |b| = B_DOVOLJEN brez kazni, nad tem kazen ((|b|-B_DOVOLJEN)/B_SIGMA)^2
# v enotah log(chi2); nad B_MAX še dodatna kazen B_KAZEN_MAX.
B_DOVOLJEN = 0.2
B_SIGMA = 0.1
B_MAX = 0.2
B_KAZEN_MAX = 100.0

# Maska Na D [Å, rest valovne dolžine]; zvezdna maska se premakne za RV.
MASKA_ZVEZDNI = (5885.0, 5900.0)  # zvezdni Na D dublet
MASKA_ISM = (5888.5, 5897.5)  # mirujoči ISM Na D

# Fiksni parametri modela
LOGG, VMIC, VMAC, VSINI = 1.0, 2.0, 3.0, 10.0
RESOLUCIJA = {"asiago": 24000, "varese": 11000}  # ločljivost R po instrumentu
RESOLUCIJA_DEFAULT = 20000

# Mreža prostih parametrov
TEFF_MREZA = [
    2700.0,
    2800.0,
    2900.0,
    3000.0,
    3100.0,
    3200.0,
    3300.0,
    3400.0,
    3500.0,
    3600.0,
    3700.0,
    3800.0,
    3900.0,
]
MONH_MREZA = [-0.8, -0.4, 0.0, 0.2, 0.4, 0.6]
NA_MREZA = [-0.3, 0.0, 0.3, 0.6, 0.9, 1.2]


# ---------------------------------------------------------------
# BRANJE PODATKOV
# ---------------------------------------------------------------
def preberi_spekter(fits_path):
  """Prebere 1D spekter iz FITS datoteke in sestavi valovno skalo.

  Zakaj: spektri so shranjeni v različnih oblikah (1D ali 2D z enim redom);
  valovna skala se rekonstruira iz FITS glave (CRVAL1, CDELT1/CD1_1, CRPIX1).

  Args:
    fits_path: pot do FITS datoteke (str).

  Returns:
    (wave, flux): valovne dolžine [Å] in SUROV fluks, obe polji numpy (float).

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


def nalozi_literaturne_rv(csv_path):
  """Prebere literaturne RV (začetna vrednost iskanja) iz CSV.

  Zakaj: iskanje RV začne pri literaturni vrednosti (GALAH DR3, sicer Gaia DR3),
  da se izognemo lokalnim minimumom χ².

  Args:
    csv_path: pot do CSV z stolpci 'ID', 'RV_GALAH [km/s]', 'RV_Gaia [km/s]'.

  Returns:
    slovar {ID (velike črke): (rv [km/s], vir 'GALAH' ali 'Gaia')}. Objekti brez
    obeh vrednosti niso v slovarju.
  """
  df = pd.read_csv(csv_path)
  slovar = {}
  for _, v in df.iterrows():
    if pd.notna(v.get("RV_GALAH [km/s]")):
      slovar[v["ID"].upper()] = (float(v["RV_GALAH [km/s]"]), "GALAH")
    elif pd.notna(v.get("RV_Gaia [km/s]")):
      slovar[v["ID"].upper()] = (float(v["RV_Gaia [km/s]"]), "Gaia")
  return slovar


class Spekter:
  """Opazovani spekter, pripravljen za fit: surov fluks v oknu FIT_OKNO.

  Zakaj: vse, kar fit potrebuje (piksli, fluks, uteži, ločljivost), je na enem
  mestu. Fluks se NE normalizira - spektri so že pred-obdelani (_c.fits), b in c
  v fitu poskrbita za morebiten zamik in naklon. Za spektre s koncnico _f, ki 
  imajo enote v fluksu se ga deli z njihovo mediano.

  Atributi:
    path: pot do FITS datoteke.
    ime: ime datoteke.
    objekt, instrument: oznaka zvezde (npr. 'A2-001') in instrumenta
      ('asiago'/'varese').
    x: valovne dolžine pikslov v FIT_OKNO [Å].
    y: SUROV fluks na teh pikslih.
    w_vse: uteži 1 za vse piksle.
    res: ločljivost R za ta instrument (za SME model).
  """

  def __init__(self, fits_path, objekt, instrument):
    """Prebere spekter in izreže okno fita.

    Args:
      fits_path: pot do FITS datoteke.
      objekt: ime objekta (npr. 'A2-001').
      instrument: 'asiago' ali 'varese' (določa ločljivost).

    Raises:
      ValueError: če spekter ne pokriva območja fita.
    """
    self.path = fits_path
    self.ime = os.path.basename(fits_path)
    self.objekt, self.instrument = objekt, instrument
    wave, flux = preberi_spekter(fits_path)

    m = (
        (wave >= OBMOCJE_BRANJE[0])
        & (wave <= OBMOCJE_BRANJE[1])
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

    # --- SAMO ZA SPEKTRE Z ZELO NIZKIM FLUKSOM (_f.fits z 1e-13) ---
    mediana_fluksa = np.nanmedian(y[m_fit])
    if 0 < mediana_fluksa < 0.01:
      y = y / mediana_fluksa

    self.x, self.y = x[m_fit], y[m_fit]
    self.w_vse = np.ones_like(self.x, dtype=float)
    self.res = RESOLUCIJA.get(instrument, RESOLUCIJA_DEFAULT)


# ---------------------------------------------------------------
# SME MODELI
# ---------------------------------------------------------------
def sintetiziraj(wave_rest, linelist, teff, monh, dna, res):
  """Sintetizira SME spekter za dane parametre (draga operacija).

  Zakaj: model se sintetizira samo enkrat na točko mreže, nato se samo premika
  po RV (np.interp). Brez pysme.solve, ker je prepočasen.

  Args:
    wave_rest: valovna mreža modela v mirovnem sistemu [Å].
    linelist: seznam črt (pysme ValdFile).
    teff: efektivna temperatura [K].
    monh: kovinskost [M/H] (dex).
    dna: odstopanje abundance Na od skalirane Sončeve vrednosti (dex).
    res: ločljivost instrumenta R (Gaussov instrumentalni profil).

  Returns:
    (wave, synth): valovne dolžine [Å] in normiran sintetični fluks (kontinuum = 1).
  """
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


class ModelCache:
  """Predpomnilnik sintetičnih modelov (v pickle datoteki na disku).

  Zakaj: sinteza je draga; vsak model (ločljivost, Teff, [M/H], dNa) se izračuna
  samo enkrat in ostane shranjen za naslednje zagone ter za normalizacija.py.

  Atributi:
    linelist: seznam črt za sintezo.
    cache_file: pot do pickle datoteke.
    wave_rest: skupna valovna mreža vseh modelov (OBMOCJE_SINTEZE, korak DLAM_MODEL).
    cache: slovar {(res, teff, monh, dna): (wave, synth)}.
    nov: število novih sintez v tem zagonu.
  """

  def __init__(self, linelist, cache_file):
    """Naloži obstoječi predpomnilnik, če obstaja.

    Args:
      linelist: seznam črt (pysme ValdFile).
      cache_file: pot do pickle datoteke za shranjevanje modelov.
    """
    self.linelist = linelist
    self.cache_file = cache_file
    self.wave_rest = np.arange(
        OBMOCJE_SINTEZE[0], OBMOCJE_SINTEZE[1], DLAM_MODEL
    )
    self.cache = {}
    if os.path.exists(cache_file):
      with open(cache_file, "rb") as f:
        self.cache = pickle.load(f)
      print(f"Naloženih {len(self.cache)} modelov iz {cache_file}")
    self.nov = 0

  def get(self, res, teff, monh, dna):
    """Vrne model; če ga ni v predpomnilniku, ga sintetizira in shrani.

    Args:
      res: ločljivost R.
      teff: efektivna temperatura [K].
      monh: kovinskost [M/H].
      dna: odstopanje abundance Na [dex].

    Returns:
      (wave, synth) v mirovnem sistemu. Po vsakih 10 novih modelih se
      predpomnilnik sproti shrani na disk.
    """
    key = (res, teff, monh, dna)
    if key not in self.cache:
      print(
          f"    [PySME sinteza] R={res}, Teff={teff:.0f}, monh={monh:+.1f},"
          f" dNa={dna:+.1f}"
      )
      self.cache[key] = sintetiziraj(
          self.wave_rest, self.linelist, teff, monh, dna, res
      )
      self.nov += 1
      if self.nov % 10 == 0:
        self.shrani()
    return self.cache[key]

  def shrani(self):
    """Shrani predpomnilnik na disk (pickle)."""
    with open(self.cache_file, "wb") as f:
      pickle.dump(self.cache, f)


# ---------------------------------------------------------------
# FIT RV
# ---------------------------------------------------------------
def chi2_pri_rv(v, x, y, w, wave_rest, synth):
  """χ² aditivnega fita modela na SUROV fluks pri dani RV.

  Model se premakne za RV in se prilagodi opazovanemu spektru z aditivnim
  zamikom b in naklonom c (utežen linearni najmanjši kvadrati):
      y ~= model + b + c * (x - x_sredina),   x_sredina = povprečje x
  Modela se NE množi (brez 'a'), da se črte umetno ne podaljšujejo.

  Zakaj: RV je tisti parameter, ki ga iščemo; b in c se za vsak RV rešita
  analitično, zato je iskanje RV samo 1D skeniranje.

  Args:
    v: RV [km/s], za katerega se računa χ².
    x: valovne dolžine opazovanega spektra [Å] (okno fita).
    y: SUROV opazovani fluks na x.
    w: uteži pikslov (0 = piksel je maskiran, 1 = uporabljen).
    wave_rest: valovna mreža modela v mirovnem sistemu [Å].
    synth: sintetični fluks modela na wave_rest.

  Returns:
    (chi2, fit, b, c): vsota uteženih kvadratov razlik, model z b in c
    (model + b + c*dx), aditivni zamik b in naklon c [fluks/Å].
  """
  model = np.interp(x / (1.0 + v / C_KMS), wave_rest, synth)
  dx = x - np.mean(x)

  A = np.vstack([np.ones_like(x), dx]).T  # stolpca: b (konstanta) in c (naklon)
  tarca = y - model  # kar mora razložiti b + c*dx

  coef, *_ = np.linalg.lstsq(A * w[:, None], tarca * w, rcond=None)
  b, c = coef[0], coef[1]

  fit = model + (b + c * dx)
  chi2 = np.sum(((y - fit) * w) ** 2)
  return chi2, fit, b, c


def skeniraj_rv(sp, model, rv_center, korak, okno, w_mask=None):
  """Izračuna χ² na mreži RV okoli rv_center.

  Zakaj: model je že sintetiziran, zato je iskanje RV samo preprosto skeniranje
  χ²(RV); minimum določa RV.

  Args:
    sp: objekt Spekter.
    model: par (wave_rest, synth) iz ModelCache.
    rv_center: središče iskanja [km/s].
    korak: korak mreže RV [km/s].
    okno: polovična širina iskanja [km/s].
    w_mask: uteži pikslov; če None, so vsi piksli uporabljeni.

  Returns:
    (rv_grid, chi2): mreža RV [km/s] in χ² na njej.
  """
  wr, syn = model
  w = sp.w_vse if w_mask is None else w_mask
  rv_grid = np.arange(rv_center - okno, rv_center + okno + korak, korak)
  chi2 = np.array([chi2_pri_rv(v, sp.x, sp.y, w, wr, syn)[0] for v in rv_grid])
  return rv_grid, chi2


def ustvari_masko_brez_na(x, v_grobi):
  """Uteži, ki izločijo zvezdni Na D dublet in mirujoči ISM Na D.

  Zakaj: Na D je zelo širok in občutljiv na dNa; za natančen RV iz kovinskih
  črt ga izločimo. ISM absorpcija je v mirovnem sistemu in je model nima.

  Args:
    x: valovne dolžine pikslov [Å] (opazovani sistem).
    v_grobi: RV zvezde [km/s]; zvezdna maska se premakne za ta RV.

  Returns:
    polje uteži: 0 znotraj zvezdne in ISM maske, sicer 1.
  """
  w = np.ones_like(x, dtype=float)
  faktor = 1.0 + v_grobi / C_KMS
  zvezdni_min, zvezdni_max = MASKA_ZVEZDNI[0] * faktor, MASKA_ZVEZDNI[1] * faktor
  ism_min, ism_max = MASKA_ISM

  maska_izrez = ((x >= zvezdni_min) & (x <= zvezdni_max)) | (
      (x >= ism_min) & (x <= ism_max)
  )
  w[maska_izrez] = 0.0
  return w


def najdi_rv(
    sp, model, rv_lit, korak_g, okno_g, korak_f, okno_f, parabola=True
):
  """2-stopenjsko iskanje RV: grobo (z Na D), nato fino (maskiran Na D).

  Zakaj: grobi korak z Na D ujame tudi velike premike (dvozvezdja); fini korak
  na kovinskih črtah da natančen RV. Isti postopek se uporablja pri izbiri
  modela (večji koraki, brez parabole) in pri končnem fitu.

  Args:
    sp: objekt Spekter.
    model: par (wave_rest, synth) iz ModelCache.
    rv_lit: literaturni RV [km/s] (središče grobega iskanja).
    korak_g, okno_g: korak in polovično okno grobe stopnje [km/s].
    korak_f, okno_f: korak in polovično okno fine stopnje [km/s].
    parabola: če True, se RV natančneje določi s parabolo skozi minimum χ².

  Returns:
    slovar z ključi: rv (RV [km/s]), a (kvadratni člen parabole, ali nan),
    chi2_min (χ² na kovinskih črtah), chi2_na (χ² z Na D, maskiran le ISM -
    edini del, ki omejuje dNa), fit (model + b + c*dx), b, c (zamik in naklon),
    w (uteži fine stopnje), rv_grid, chi2_grid (fina mreža RV in χ²),
    na_robu (True, če je minimum na robu fine mreže).
  """
  wr, syn = model

  # 1. STOPNJA: grobi RV (celoten spekter z Na D, široko okno)
  rv_g1, chi2_g1 = skeniraj_rv(sp, model, rv_lit, korak_g, okno_g)
  v_grobi = rv_g1[np.argmin(chi2_g1)]

  # 2. STOPNJA: fini RV (samo okoliške kovinske črte)
  w_kovinske = ustvari_masko_brez_na(sp.x, v_grobi)
  rv_g2, chi2_g2 = skeniraj_rv(
      sp, model, v_grobi, korak_f, okno_f, w_mask=w_kovinske
  )

  i_min = int(np.argmin(chi2_g2))
  a = np.nan
  rv_sme = rv_g2[i_min]
  if parabola:
    sl = slice(max(i_min - 8, 0), min(i_min + 9, len(rv_g2)))
    a, b_fit, _ = np.polyfit(rv_g2[sl], chi2_g2[sl], 2)
    if a > 0:
      rv_sme = -b_fit / (2 * a)

  # Končni model, zamik b in naklon c pri najdenem RV (kovinske črte)
  chi2_min, fit, b_offset, c_slope = chi2_pri_rv(
      rv_sme, sp.x, sp.y, w_kovinske, wr, syn
  )

  # χ² z Na D (maskiran le mirujoči ISM): edini del, ki omejuje dNa
  w_ism = np.ones_like(sp.x, dtype=float)
  w_ism[(sp.x >= MASKA_ISM[0]) & (sp.x <= MASKA_ISM[1])] = 0.0
  chi2_na = chi2_pri_rv(rv_sme, sp.x, sp.y, w_ism, wr, syn)[0]

  return dict(
      rv=rv_sme,
      a=a,
      chi2_min=chi2_min,
      chi2_na=chi2_na,
      fit=fit,
      b=b_offset,
      c=c_slope,
      w=w_kovinske,
      rv_grid=rv_g2,
      chi2_grid=chi2_g2,
      na_robu=i_min in (0, len(rv_g2) - 1),
  )


# ---------------------------------------------------------------
# IZBIRA MODELA IN FIT SPEKTRA
# ---------------------------------------------------------------
def kazen_b(b):
  """Kazen za prevelik |b| pri izbiri modela (v enotah log χ²).

  Zakaj: hladni, kovinsko bogati modeli imajo nizko povprečje; velik b jih
  'dvigne' do ravne črte in fit je lažno dober. Kazen to prepreči.

  Args:
    b: aditivni zamik iz fita.

  Returns:
    0 do |b| = B_DOVOLJEN, nad tem kvadratna kazen ((|b|-B_DOVOLJEN)/B_SIGMA)^2,
    ob preseženem B_MAX še dodatnih B_KAZEN_MAX.
  """
  presezek = max(abs(b) - B_DOVOLJEN, 0.0)
  kazen = (presezek / B_SIGMA) ** 2
  if abs(b) > B_MAX:
    kazen += B_KAZEN_MAX
  return kazen


def izberi_model_objekta(spektri, rv_lit, modeli):
  """Izbere najboljši model (Teff, [M/H], dNa) iz mreže za en objekt.

  Zakaj: vsi spektri istega objekta imajo iste zvezdne parametre, zato se model
  izbere enkrat na objekt. Za vsak model in vsak izbrani spekter se RV določi
  z istim postopkom kot pri končnem fitu; ocena modela je povprečje
  log(χ² z Na D) + kazen_b(b). Na D je v χ² zato, ker sicer dNa ni omejen.

  Args:
    spektri: seznam objektov Spekter istega objekta.
    rv_lit: literaturni RV [km/s].
    modeli: objekt ModelCache.

  Returns:
    ((teff, monh, dna), ocena) najboljšega modela (nižja ocena = boljši).
  """
  izbor = (
      spektri
      if len(spektri) <= N_IZBOR
      else [
          spektri[i]
          for i in np.linspace(0, len(spektri) - 1, N_IZBOR).astype(int)
      ]
  )

  rezultati = {}
  for teff, monh, dna in itertools.product(TEFF_MREZA, MONH_MREZA, NA_MREZA):
    skupaj = 0.0
    for sp in izbor:
      model = modeli.get(sp.res, teff, monh, dna)
      r = najdi_rv(
          sp,
          model,
          rv_lit,
          IZB_KORAK_GROBO,
          RV_OKNO_GROBO,
          IZB_KORAK_FINO,
          IZB_OKNO_FINO,
          parabola=False,
      )
      skupaj += np.log(r["chi2_na"]) + kazen_b(r["b"])
    rezultati[(teff, monh, dna)] = skupaj / len(izbor)

  best = min(rezultati, key=rezultati.get)
  return best, rezultati[best]


def fit_spekter(sp, rv_lit, vir, par, modeli):
  """Končni fit RV, b in c za en spekter z izbranim modelom objekta.

  Zakaj: to je glavni rezultat skripte - RV in napaka, ter b in c, ki ju
  normalizacija.py uporabi za kontinuum C(x) = 1 + b + c*(x - x_sredina).

  Args:
    sp: objekt Spekter.
    rv_lit: literaturni RV [km/s].
    vir: vir literaturnega RV ('GALAH' ali 'Gaia').
    par: (teff, monh, dna) izbranega modela objekta.
    modeli: objekt ModelCache.

  Returns:
    slovar (ena vrstica tabele): objekt, instrument, spekter, rv_lit, vir,
    rv_sme, e_rv_sme, razlika (rv_sme - rv_lit), b_offset, c_slope, teff, monh,
    dna, sigma (rms razlike fit-spekter v enotah fluksa), na_robu.
  """
  teff, monh, dna = par
  model = modeli.get(sp.res, teff, monh, dna)

  r = najdi_rv(
      sp,
      model,
      rv_lit,
      RV_KORAK_GROBO,
      RV_OKNO_GROBO,
      RV_KORAK_FINO,
      RV_OKNO_FINO,
      parabola=True,
  )
  rv_sme, a = r["rv"], r["a"]
  w_kovinske = r["w"]

  dof = int(w_kovinske.sum()) - 3  # prostostne stopnje: piksli - (RV, b, c)
  sigma = np.sqrt(r["chi2_min"] / dof) if dof > 0 else np.nan
  e_rv = sigma / np.sqrt(a) if a > 0 else np.nan

  rez = dict(
      objekt=sp.objekt,
      instrument=sp.instrument,
      spekter=sp.ime,
      rv_lit=rv_lit,
      vir=vir,
      rv_sme=rv_sme,
      e_rv_sme=e_rv,
      razlika=rv_sme - rv_lit,
      b_offset=r["b"],
      c_slope=r["c"],
      teff=teff,
      monh=monh,
      dna=dna,
      sigma=sigma,
      na_robu=r["na_robu"],
  )

  if RISI_SLIKE:
    narisi(sp, r["fit"], r["rv_grid"], r["chi2_grid"], w_kovinske, rez)

  return rez


# ---------------------------------------------------------------
# IZPIS
# ---------------------------------------------------------------
def narisi(sp, fit, rv_grid, chi2, w_mask, rez):
  """Nariše diagnostično sliko fita enega spektra in jo shrani v SLIKE_DIR.

  Zakaj: vizualna kontrola ujemanja modela s spektrom in oblike χ²(RV).
  Zgoraj: surov spekter in model (z b in c); spodaj: χ² kovinskih črt po RV.

  Args:
    sp: objekt Spekter (x, y = surov fluks).
    fit: model + b + c*dx na pikslih sp.x.
    rv_grid: mreža RV fine stopnje [km/s].
    chi2: χ² (kovinske črte) na rv_grid.
    w_mask: uteži fine stopnje (0 = maskiran Na D).
    rez: slovar iz fit_spekter (rv_sme, e_rv_sme, rv_lit, vir, b_offset,
      c_slope, teff, monh, ...).
  """
  fig, ax = plt.subplots(
      2, 1, figsize=(12, 7.5), gridspec_kw=dict(height_ratios=[2, 1])
  )

  # Zgornji panel: spekter in model
  ax[0].plot(sp.x, sp.y, "k-", lw=1.0, label="Opazovano (surov fluks)")
  ax[0].plot(
      sp.x,
      fit,
      "r--",
      lw=1.5,
      label=(
          f"SME (Teff={rez['teff']:.0f}, monh={rez['monh']:+.1f}, "
          f"b={rez['b_offset']:+.3f}, c={rez['c_slope'] * 100:+.3f}/100Å)"
      ),
  )

  # Maskirano območje Na D
  m_mask = w_mask == 0.0
  if m_mask.sum() > 0:
    ax[0].axvspan(
        sp.x[m_mask].min(),
        sp.x[m_mask].max(),
        color="orange",
        alpha=0.15,
        label="Maskirano Na D (fittane le kovinske črte)",
    )

  ax[0].set_xlim(*FIT_OKNO)
  ax[0].set_title(
      f"{rez['objekt']} – {rez['spekter']} | RV_SME = {rez['rv_sme']:.2f} ±"
      f" {rez['e_rv_sme']:.2f} km/s"
  )
  ax[0].legend(loc="lower left", fontsize=9)
  ax[0].grid(alpha=0.3)

  # Spodnji panel: χ² po RV (samo kovinske črte)
  ax[1].plot(rv_grid, chi2, "b.-")
  ax[1].axvline(
      rez["rv_lit"],
      color="g",
      ls="--",
      label=f"Lit ({rez['vir']}): {rez['rv_lit']:.2f}",
  )
  ax[1].axvline(
      rez["rv_sme"],
      color="r",
      ls="--",
      label=f"SME (kovinske): {rez['rv_sme']:.2f}",
  )
  ax[1].set_xlabel("RV (km/s)")
  ax[1].set_ylabel(r"$\chi^2$ (kovinske črte)")
  ax[1].legend(fontsize=9)
  ax[1].grid(alpha=0.3)

  plt.tight_layout()
  ime = rez["spekter"].replace(".fits", "")
  plt.savefig(
      os.path.join(SLIKE_DIR, f"{rez['objekt']}_{rez['instrument']}_{ime}.png"),
      dpi=90,
  )
  plt.close()


def zapisi_txt(rezultati, path):
  """Zapiše tabelo rezultatov (ena vrstica na spekter) v besedilno datoteko.

  Zakaj: tabelo prebereta normalizacija.py (RV, model, b, c) in analize.

  Args:
    rezultati: seznam slovarjev iz fit_spekter.
    path: izhodna pot (rv_sme_rezultati.txt).
  """
  df = pd.DataFrame(rezultati)
  with open(path, "w") as f:
    f.write(
        "# RV iz okoliških kovinskih črt (aditivni b in c na surovem fluksu)\n"
    )
    f.write("# b_offset = aditivni zamik; c_slope = naklon [fluks/Å]\n")
    f.write(
        df.to_string(
            index=False,
            formatters={
                "rv_sme": "{:.2f}".format,
                "e_rv_sme": "{:.2f}".format,
                "razlika": "{:+.2f}".format,
                "b_offset": "{:+.4f}".format,
                "c_slope": "{:+.6f}".format,
            },
        )
    )
    f.write("\n")


def zapisi_modele(modeli_obj, path):
  """Zapiše izbrane parametre modela (Teff, [M/H], dNa) za vsak objekt.

  Args:
    modeli_obj: slovar {objekt: ((teff, monh, dna), število_spektrov)}.
    path: izhodna pot (rv_sme_modeli_objektov.txt).
  """
  with open(path, "w") as f:
    f.write("# Izbrani parametri modela po objektih\n")
    f.write(f"{'objekt':8s} {'Teff':>6s} {'monh':>6s} {'dNa':>6s}\n")
    for obj, (par, _) in sorted(modeli_obj.items()):
      f.write(f"{obj:8s} {par[0]:6.0f} {par[1]:+6.1f} {par[2]:+6.1f}\n")


# ---------------------------------------------------------------
# GLAVNI DEL
# ---------------------------------------------------------------
def main():
  """Obdela vse spektre: izbere model objekta, določi RV, b, c in zapiše izhode.

  Zakaj: vstopna točka skripte. Po vsakem objektu se rezultati sproti zapišejo,
  da se ob prekinitvi nič ne izgubi.
  """
  if RISI_SLIKE:
    os.makedirs(SLIKE_DIR, exist_ok=True)
  rv_slovar = nalozi_literaturne_rv(RV_CSV)
  modeli = ModelCache(ValdFile(VALD_PATH), CACHE_FILE)

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

  print(
      f"Objektov: {len(po_objektih)}, spektrov:"
      f" {sum(len(v) for v in po_objektih.values())}"
  )

  rezultati, modeli_obj = [], {}
  for k, (objekt, spektri) in enumerate(sorted(po_objektih.items()), 1):
    rv_lit, vir = rv_slovar[objekt]
    print(
        f"\n[{k}/{len(po_objektih)}] {objekt}: {len(spektri)} spektrov,"
        f" RV_lit={rv_lit:.2f} ({vir})"
    )

    par, _ = izberi_model_objekta(spektri, rv_lit, modeli)
    modeli_obj[objekt] = (par, len(spektri))
    print(
        f"    Izbran model: Teff={par[0]:.0f}, monh={par[1]:+.1f},"
        f" dNa={par[2]:+.1f}"
    )

    for sp in spektri:
      try:
        r = fit_spekter(sp, rv_lit, vir, par, modeli)
        rezultati.append(r)
        opoz = "  !! NA ROBU" if r["na_robu"] else ""
        print(
            f"    {sp.ime}: RV={r['rv_sme']:.2f}±{r['e_rv_sme']:.2f} "
            f"b={r['b_offset']:+.3f} diff={r['razlika']:+.2f}{opoz}"
        )
      except Exception as e:
        preskoceni.append((sp.path, str(e)))
        print(f"    {sp.ime}: PRESKOČEN: {e}")

    modeli.shrani()
    zapisi_txt(rezultati, IZHOD_TXT)
    zapisi_modele(modeli_obj, MODELI_TXT)

  print(f"\nKončano! Rezultati so v: {IZHOD_TXT}")


if __name__ == "__main__":
  main()