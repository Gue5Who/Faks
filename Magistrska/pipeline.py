"""
Fit absorpcijskih komponent na natrijevem dubletu Na I D (D2 5889.95 A, D1 5895.92 A)
z Voigtovimi profili: zvezdna komponenta + dinamično število ISM oblakov (BIC).

MODEL
    Skupni prenos sevanja (kontinuum = 1):
        F(lam) = G_inst  *  exp( -tau_zvezda(lam) - sum_i tau_ISM,i(lam) )
    kjer je * konvolucija s Gaussovim instrumentalnim profilom (FWHM = lam / R).
    Konvolucija se naredi nad TOKOM (exp(-tau)), ne nad tau, ker instrument
    zgladi tok. To je pomembno pri nasičenih ISM črtah.

    Vsaka komponenta je dublet: obe črti (D2, D1) imata ISTI radialni hitrosti v
    in Dopplerjevo širino b. Optični globini tau_D2 in tau_D1 sta PROSTI (razmerje
    ni zaklenjeno na 2:1). tau je optična globina v centru nekonvolviranega profila.
      - ISM: dušenje je naravno (radiacijsko), b in tau omejena (glej NASTAVITVE).
      - Zvezda: b v [10, 50] km/s, dušenje 'a' je prosto (tlačno razširjena krila
        Na D v hladnih zvezdah), v je omejena na rv_sme +- VSTAR_MEJA (rv_sme je samo
        začetni približek, ne togo sidro).

PODATKI
    Za vsak spekter iz figs/sme_rv/rv_sme_rezultati.txt se vzame surov FITS, ga
    normalizira z b_offset in c_slope (isti postopek kot normalizacija.py, kontinuum
    C = 1 + b + c*dx) in fita okno FIT_NA.

ŠTEVILO ISM OBLAKOV
    Modeli z N_ISM = 1 ... N_ISM_MAX se prilegajo zaporedno (vsak naslednji se začne
    iz rešitve prejšnjega + nova komponenta na najboljši poziciji skeniranja).
    Izbira po BIC = chi2 + k * ln(M); dodatna komponenta se sprejme le, če
    BIC pade za več kot DELTA_BIC (sicer ostane enostavnejši model).
    Napake meritev niso znane, zato je sigma za chi2 enaka za vse modele istega
    spektra in se oceni iz ostanka najkompleksnejšega modela (sigma^2 = RSS/(M-k)),
    z najmanjšo vrednostjo iz šuma sosednjih pikslov. Neopisane zvezdne črte tako
    ne bodo same po sebi 'upravičile' dodatnih ISM oblakov.

IZHODI (mapa Rezultati/<objekt>/<instrument>/)
    <ime>_fit.png   - opazovanje, skupni fit, zvezda in vsak ISM oblak posebej
    <ime>_data.txt  - parametri z napakami, tabela BIC in stolpci fita
    Rezultati/ism_rezultati.txt - skupna tabela (ena vrstica na komponento).

Uporaba:
    python pipeline.py                  # vsi spektri iz tabele
    python pipeline.py A2-342 A2-001    # samo izbrani objekti
"""

import os

# Brez večnitnosti znotraj procesa: vzporednost je po spektrih (multiprocessing).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import glob
import multiprocessing as mp
import sys

import matplotlib

matplotlib.use("Agg")  # brez okna; mora biti pred uvozom pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import wofz

# Branje FITS, okno in normalizacija ostanejo ENAKI kot v normalizacija.py
from normalizacija import (
    RESOLUCIJA,
    RESOLUCIJA_DEFAULT,
    izrezi_okno,
    normaliziraj_bc,
    preberi_spekter,
)

# ---------------------------------------------------------------
# KONSTANTE
# ---------------------------------------------------------------
C_KMS = 299792.458  # hitrost svetlobe [km/s]
W_D2 = 5889.950  # rest valovna dolžina Na I D2 [A]
W_D1 = 5895.924  # rest valovna dolžina Na I D1 [A]
W_SRED = 5893.0  # sredina dubleta [A] (za instrumentalno širino)
GAMMA_NAT = 6.14e7  # naravna sevalna širina obeh črt [1/s]
FWHM_V_SIGMA = 2.3548  # FWHM = 2.3548 * sigma pri Gaussu

# ---------------------------------------------------------------
# NASTAVITVE
# ---------------------------------------------------------------
REZULTATI_TXT = "figs/sme_rv/rv_sme_rezultati.txt"  # tabela iz sme_rv_vsi.py
SPEKTRI_GLOB = "Paper_II/*/*/*.fits"
SPEKTRI_KOREN = "Paper_II"
IZHOD_DIR = "Rezultati"
TABELA_TXT = os.path.join(IZHOD_DIR, "ism_rezultati.txt")
N_PROCESOV = max(1, (os.cpu_count() or 2) - 1)  # vzporedna obdelava spektrov

FIT_NA = (5876.0, 5909.0)  # okno fita okoli Na D [A]
DLAM_FIN = 0.01  # korak fine mreže modela [A] (pred konvolucijo)

# Število ISM oblakov (BIC)
N_ISM_MAX = 3  # največ ISM komponent
DELTA_BIC = 10.0  # dodatna komponenta mora znižati BIC za več kot toliko
SEP_FAKTOR = 0.5  # min. razmik dveh ISM: SEP_FAKTOR * dv_inst

# Meje ISM komponent
V_ISM = (-100.0, 100.0)  # radialna hitrost ISM [km/s]
B_ISM = (0.3, 9.0)  # Dopplerjeva širina ISM [km/s]
TAU_ISM = (0.02, 15.0)  # optična globina v centru (D2 in D1 vsaka posebej)

# Meje zvezdne komponente
VSTAR_MEJA = 4.0  # zvezdni v v [rv_sme - X, rv_sme + X] [km/s]
B_ZVEZDA = (1.0, 20.0)  # Dopplerjeva širina zvezde [km/s]
A_ZVEZDA = (0.01, 4)  # parameter dušenja Voigta zvezde
LGTAU_ZVEZDA = (0.5, 3.5)  # log10 optične globine zvezde

# Iskanje začetnih vrednosti in optimizacija
V_SKEN_KORAK = 5.0  # korak skeniranja položaja nove ISM komponente [km/s]
N_ZACETKOV = 2  # št. začetnih položajev nove komponente, ki jih fitamo do konca
MAX_NFEV = 150  # največ evalvacij modela na en fit
KAZEN_V_ISM = 10.0  # utež kazni, če kumulativna ISM hitrost preseže V_ISM[1]

# Sistematske napake (enako kot v prejšnji različici)
SYS_V_DELITELJ = 10.0  # napaka hitrosti = dv_inst / 10
SYS_B_DELITELJ = 20.0  # napaka širine b = dv_inst / 20
SYS_TAU = 0.02  # napaka optične globine ISM zaradi kontinuuma


# ---------------------------------------------------------------
# VOIGTOV PROFIL
# ---------------------------------------------------------------
def voigt_h(a, u):
    """Voigtova funkcija H(a, u) = Re[w(u + i*a)] prek Faddeeve funkcije.

    Zakaj: optična globina absorpcijske črte s Gaussovim jedrom (Doppler) in
    Lorentzovimi krili (naravno/tlačno dušenje) je Voigtov profil.

    Args:
        a: parameter dušenja (brezdimenzijski), a = Gamma / (4*pi*dnu_D); skalar.
        u: brezdimenzijski odmik od centra, (lam - lam_c) / dlam_D; skalar ali polje.

    Returns:
        H(a, u) (brezdimenzijsko), enake oblike kot u. H(a, 0) = exp(a^2) erfc(a).
    """
    return np.real(wofz(u + 1j * a))


def tau_voigt(lam, lam0, v, b, a, tau0):
    """Optična globina ene absorpcijske črte kot Voigtov profil.

    Zakaj: osnovni gradnik modela; profil je normiran tako, da je v centru
    optična globina natanko tau0 (lažja interpretacija kot ekvivalentna širina).

    Args:
        lam: valovne dolžine [A] (polje).
        lam0: rest valovna dolžina črte [A] (W_D2 ali W_D1).
        v: radialna hitrost komponente [km/s].
        b: Dopplerjeva širina [km/s] (Gauss exp(-u^2), u = dlam/dlam_D).
        a: parameter dušenja Voigta (brezdimenzijski).
        tau0: optična globina v centru profila (brezdimenzijska).

    Returns:
        polje optične globine tau(lam) enake oblike kot lam.
    """
    lam_c = lam0 * (1.0 + v / C_KMS)
    dlam_d = lam_c * b / C_KMS
    u = (lam - lam_c) / dlam_d
    return tau0 * voigt_h(a, u) / voigt_h(a, 0.0)


def damping_naravni(lam0, b):
    """Parameter dušenja Voigta zaradi naravne (sevalne) širine črte.

    Zakaj: ISM plin je redek, zato je dušenje določeno le z naravno širino
    (ni prosti parameter); vendar je odvisno od b, zato se računa sproti.

    Args:
        lam0: rest valovna dolžina črte [A].
        b: Dopplerjeva širina [km/s].

    Returns:
        a = Gamma * lam0 / (4*pi*b) (brezdimenzijski; za b ~ 3 km/s okoli 1e-3).
    """
    return GAMMA_NAT * (lam0 * 1e-8) / (4.0 * np.pi * b * 1e5)


class ModelNaD:
    """Model prenosa sevanja na Na D: Voigtovi profili + instrumentalna konvolucija.

    Zakaj: instrumentalna širina (12-27 km/s) je večja od ISM črt (b < 8 km/s),
    zato je treba model izračunati na fini mreži in šele nato konvolvirati s
    Gaussom, nato pa interpolirati na piksle spektra.

    Atributi:
        R: ločljivost instrumenta lam / dlam.
        dv_inst: instrumentalna ločljivost c / R [km/s] (FWHM).
        sigma_lam: sigma instrumentalnega Gaussa [A].
        grid: fina valovna mreža [A] (okno fita + rob za konvolucijo).
        kernel: normiran Gaussov konvolucijski kernel na koraku DLAM_FIN.
    """

    def __init__(self, lam_lo, lam_hi, R):
        """Pripravi fino mrežo in instrumentalni kernel.

        Args:
            lam_lo: spodnja meja okna fita [A].
            lam_hi: zgornja meja okna fita [A].
            R: ločljivost instrumenta (npr. 24000 Asiago, 11000 Varese).
        """
        self.R = R
        self.dv_inst = C_KMS / R
        self.sigma_lam = (W_SRED / R) / FWHM_V_SIGMA
        pad = 5.0 * self.sigma_lam + 0.5  # rob, da konvolucija ne ukrivi roba okna
        self.grid = np.arange(lam_lo - pad, lam_hi + pad, DLAM_FIN)
        nk = int(np.ceil(4.5 * self.sigma_lam / DLAM_FIN))
        kx = np.arange(-nk, nk + 1) * DLAM_FIN
        k = np.exp(-0.5 * (kx / self.sigma_lam) ** 2)
        self.kernel = k / k.sum()

    def _dodaj_ism(self, tau, komp):
        """Prišteje optično globino ene ISM komponente (D2 + D1) polju tau.

        Zakaj: ISM črte so ozke, zato se profil izračuna samo v lokalnem oknu okoli
        centra (+-10 Dopplerjevih širin, vsaj +-0.3 A); zunaj je tau zanemarljiv,
        računanje wofz pa je tako bistveno hitrejše.

        Args:
            tau: polje optične globine na self.grid (spremeni se na mestu).
            komp: (v [km/s], b [km/s], tau_D2, tau_D1) ISM komponente.
        """
        v, b, t2, t1 = komp
        for lam0, t0 in ((W_D2, t2), (W_D1, t1)):
            lam_c = lam0 * (1.0 + v / C_KMS)
            pol = max(0.3, 10.0 * lam_c * b / C_KMS)
            i0 = np.searchsorted(self.grid, lam_c - pol)
            i1 = np.searchsorted(self.grid, lam_c + pol)
            if i1 > i0:
                tau[i0:i1] += tau_voigt(
                    self.grid[i0:i1], lam0, v, b, damping_naravni(lam0, b), t0
                )

    def spekter(self, x, zvezda=None, ism=()):
        """Izračuna konvolviran prenos F(lam) na valovnih dolžinah x.

        Zakaj: skupni model za fit in tudi posamezne komponente za sliko (z
        zvezda=None ali ism=() se dobi samo ena komponenta).

        Args:
            x: valovne dolžine opazovanih pikslov [A].
            zvezda: (v [km/s], b [km/s], a, tau_D2, tau_D1) ali None brez zvezde.
            ism: seznam ISM komponent (v [km/s], b [km/s], tau_D2, tau_D1).

        Returns:
            polje prenosa (0..1, kontinuum = 1) na x.
        """
        tau = np.zeros_like(self.grid)
        if zvezda is not None:
            v, b, a, t2, t1 = zvezda
            tau += tau_voigt(self.grid, W_D2, v, b, a, t2)
            tau += tau_voigt(self.grid, W_D1, v, b, a, t1)
        for komp in ism:
            self._dodaj_ism(tau, komp)
        f = np.convolve(np.exp(-tau), self.kernel, mode="same")
        return np.interp(x, self.grid, f)


# ---------------------------------------------------------------
# PARAMETRI FITA
# ---------------------------------------------------------------
# Notranji vektor p za N ISM komponent:
#   [v_zv, b_zv, a_zv, lg_tau_D2_zv, lg_tau_D1_zv,        (5, zvezda)
#    v_ISM1, dv_ISM2, ..., dv_ISMN,                        (N, hitrosti)
#    b_ISM1, tau_D2_ISM1, tau_D1_ISM1, b_ISM2, ...]        (3N, širine in globine)
# ISM hitrosti so podane kot v1 in zaporedni razmiki dv_j >= SEP_FAKTOR*dv_inst,
# zato so oblaki urejeni po hitrosti in se ne morejo zliti (trda meja fita).
N_ZVEZDA = 5  # število parametrov zvezde


def razcleni_p(p, n):
    """Notranji vektor parametrov pretvori v fizikalne komponente.

    Zakaj: optimizator dela z notranjim vektorjem (log tau za zvezdo, razmiki
    hitrosti za ISM), model pa potrebuje fizikalne vrednosti.

    Args:
        p: notranji vektor parametrov (dolžina 5 + 4*n).
        n: število ISM komponent.

    Returns:
        (zvezda, ism): zvezda = (v, b, a, tau_D2, tau_D1);
        ism = seznam (v, b, tau_D2, tau_D1), urejen po naraščajoči hitrosti.
    """
    zv = (p[0], p[1], p[2], 10.0 ** p[3], 10.0 ** p[4])
    v = np.cumsum(p[N_ZVEZDA : N_ZVEZDA + n])  # v1, v1+dv2, ...
    ism = []
    for i in range(n):
        o = N_ZVEZDA + n + 3 * i
        ism.append((v[i], p[o], p[o + 1], p[o + 2]))
    return zv, ism


def sestavi_p(p_zvezda, ism):
    """Sestavi notranji vektor parametrov iz zvezde in seznama ISM komponent.

    Zakaj: začetne vrednosti (iz skeniranja ali prejšnjega fita) so podane
    fizikalno; za optimizator jih je treba pretvoriti v notranjo obliko.

    Args:
        p_zvezda: notranji vektor zvezde (5 elementov: v, b, a, lg_tau_D2, lg_tau_D1).
        ism: seznam ISM komponent (v, b, tau_D2, tau_D1); vrstni red je poljuben.

    Returns:
        notranji vektor p (dolžina 5 + 4*len(ism)); ISM urejene po hitrosti.
    """
    ism = sorted(ism, key=lambda c: c[0])
    p = list(p_zvezda)
    for i, c in enumerate(ism):
        p.append(c[0] if i == 0 else c[0] - ism[i - 1][0])
    for c in ism:
        p += [c[1], c[2], c[3]]
    return np.array(p, dtype=float)


def meje_p(n, rv_sme, dv_inst):
    """Meje in tipične velikosti parametrov za fit z n ISM komponentami.

    Zakaj: astrofizikalne meje (b, tau, razmik oblakov) so podane kot meje
    optimizatorja; x_scale pa pomaga, ker so parametri različnih redov velikosti.

    Args:
        n: število ISM komponent.
        rv_sme: izmerjeni RV zvezde iz SME [km/s] (središče mej zvezdne hitrosti).
        dv_inst: instrumentalna ločljivost c/R [km/s] (določa min. razmik oblakov).

    Returns:
        (lo, hi, x_scale): spodnje meje, zgornje meje in tipične velikosti (polja).
    """
    sep = SEP_FAKTOR * dv_inst
    lo = [rv_sme - VSTAR_MEJA, B_ZVEZDA[0], A_ZVEZDA[0], LGTAU_ZVEZDA[0], LGTAU_ZVEZDA[0]]
    hi = [rv_sme + VSTAR_MEJA, B_ZVEZDA[1], A_ZVEZDA[1], LGTAU_ZVEZDA[1], LGTAU_ZVEZDA[1]]
    xs = [5.0, 5.0, 0.5, 0.5, 0.5]
    if n > 0:
        lo.append(V_ISM[0])
        hi.append(V_ISM[1])
        xs.append(5.0)
        for _ in range(n - 1):
            lo.append(sep)
            hi.append(V_ISM[1] - V_ISM[0])
            xs.append(5.0)
        for _ in range(n):
            lo += [B_ISM[0], TAU_ISM[0], TAU_ISM[0]]
            hi += [B_ISM[1], TAU_ISM[1], TAU_ISM[1]]
            xs += [1.0, 1.0, 1.0]
    return np.array(lo), np.array(hi), np.array(xs)


def imena_p(n):
    """Imena notranjih parametrov (za izpis parametrov na mejah).

    Args:
        n: število ISM komponent.

    Returns:
        seznam imen v istem vrstnem redu kot vektor p.
    """
    imena = ["v_zvezda", "b_zvezda", "a_zvezda", "lg_tau_D2_zvezda", "lg_tau_D1_zvezda"]
    if n > 0:
        imena.append("v_ISM1")
        imena += [f"dv_ISM{i + 1}" for i in range(1, n)]
        for i in range(n):
            imena += [f"b_ISM{i + 1}", f"tau_D2_ISM{i + 1}", f"tau_D1_ISM{i + 1}"]
    return imena


def fizicni_vektor(p, n):
    """Fizikalni vektor [v, b, a, tau_D2, tau_D1 zvezde; v, b, tau_D2, tau_D1 vsakega ISM].

    Zakaj: poročamo fizikalne parametre (ne razmikov in logaritmov), napake pa se
    iz notranjih parametrov prenesejo prek Jacobijana te preslikave.

    Args:
        p: notranji vektor parametrov.
        n: število ISM komponent.

    Returns:
        polje dolžine 5 + 4*n.
    """
    zv, ism = razcleni_p(p, n)
    out = list(zv)
    for c in ism:
        out += list(c)
    return np.array(out, dtype=float)


def jakobijan_fizicni(p, n):
    """Numerični Jacobijan preslikave notranji -> fizikalni parametri.

    Args:
        p: notranji vektor parametrov.
        n: število ISM komponent.

    Returns:
        matrika (len(fizicni) x len(p)).
    """
    f0 = fizicni_vektor(p, n)
    J = np.zeros((len(f0), len(p)))
    for j in range(len(p)):
        h = 1e-6 * max(1.0, abs(p[j]))
        pp = p.copy()
        pp[j] += h
        J[:, j] = (fizicni_vektor(pp, n) - f0) / h
    return J


# ---------------------------------------------------------------
# FIT
# ---------------------------------------------------------------
def ostanek(p, n, model, x, y):
    """Vektor ostankov (opazovano - model) za least_squares.

    Zakaj: zadnji element je kazen, če kumulativna hitrost najvišjega ISM oblaka
    preseže V_ISM[1] (meje na razmikih tega same ne zagotavljajo). Dolžina
    vektorja je zato vedno M + 1.

    Args:
        p: notranji vektor parametrov.
        n: število ISM komponent.
        model: objekt ModelNaD.
        x: valovne dolžine pikslov [A].
        y: normaliziran opazovani fluks.

    Returns:
        polje (M + 1) ostankov; zadnji element je kazen (0, če so meje spoštovane).
    """
    zv, ism = razcleni_p(p, n)
    r = y - model.spekter(x, zv, ism)
    presezek = max(0.0, float(np.sum(p[N_ZVEZDA : N_ZVEZDA + n])) - V_ISM[1])
    return np.append(r, KAZEN_V_ISM * presezek)


def prilagodi(model, x, y, p0, n, rv_sme):
    """Zažene optimizacijo (trf z mejami) iz začetnih parametrov p0.

    Args:
        model: objekt ModelNaD.
        x: valovne dolžine pikslov [A].
        y: normaliziran opazovani fluks.
        p0: začetni notranji vektor parametrov (dolžina 5 + 4*n).
        n: število ISM komponent.
        rv_sme: RV zvezde iz SME [km/s] (določa meje zvezdne hitrosti).

    Returns:
        slovar: p (rešitev), rss (vsota kvadratov ostankov brez kazni), jac
        (Jacobijan ostankov), nfev (št. evalvacij), n (št. ISM komponent).
    """
    lo, hi, xs = meje_p(n, rv_sme, model.dv_inst)
    p0 = np.clip(p0, lo, hi)
    res = least_squares(
        ostanek, p0, bounds=(lo, hi), x_scale=xs, diff_step=1e-4,
        max_nfev=MAX_NFEV, args=(n, model, x, y),
    )
    M = len(x)
    return dict(
        p=res.x, rss=float(np.sum(res.fun[:M] ** 2)), jac=res.jac,
        nfev=res.nfev, n=n,
    )


def predfit_zvezde(model, x, y, rv_sme):
    """Fit samo zvezdne komponente (brez ISM) za dober začetni približek.

    Zakaj: zvezdna črta je široka in globoka; brez dobrega začetka bi skeniranje
    položaja ISM oblaka merilo napačen ostanek. Poskusita se dva začetna
    približka (šibkejše in močnejše dušenje), obdrži se boljši.

    Args:
        model: objekt ModelNaD.
        x: valovne dolžine pikslov [A].
        y: normaliziran opazovani fluks.
        rv_sme: RV zvezde iz SME [km/s] (začetni približek zvezdne hitrosti).

    Returns:
        notranji vektor zvezde (5 elementov: v, b, a, lg_tau_D2, lg_tau_D1).
    """
    zacetki = [
        [rv_sme, 20.0, 0.1, 1.5, 1.2],
        [rv_sme, 20.0, 1.0, 2.5, 2.2],
    ]
    najboljsi = None
    for z in zacetki:
        r = prilagodi(model, x, y, np.array(z), 0, rv_sme)
        if najboljsi is None or r["rss"] < najboljsi["rss"]:
            najboljsi = r
    return najboljsi["p"][:N_ZVEZDA]


def dodaj_komponento(model, x, y, p_zvezda, ism, rv_sme):
    """Doda eno ISM komponento: skeniranje položaja, nato polni fit.

    Zakaj: fit ISM hitrosti ima veliko lokalnih minimumov (ozke črte). Zato se
    nova komponenta najprej postavi na mrežo hitrosti (korak V_SKEN_KORAK) pri
    fiksnih ostalih parametrih, nato se najboljši položaji (vsaj 15 km/s narazen)
    popolnoma fitajo, obdrži se najboljši. Upošteva se min. razmik do obstoječih
    oblakov.

    Args:
        model: objekt ModelNaD.
        x: valovne dolžine pikslov [A].
        y: normaliziran opazovani fluks.
        p_zvezda: notranji vektor zvezde (5 elementov) iz prejšnjega fita.
        ism: seznam obstoječih ISM komponent (v, b, tau_D2, tau_D1).
        rv_sme: RV zvezde iz SME [km/s].

    Returns:
        slovar rezultata fita z n = len(ism) + 1 komponentami, ali None, če ni
        nobenega dovoljenega položaja za novo komponento.
    """
    sep = SEP_FAKTOR * model.dv_inst
    zv = (p_zvezda[0], p_zvezda[1], p_zvezda[2], 10.0 ** p_zvezda[3], 10.0 ** p_zvezda[4])

    kandidati = []
    for v in np.arange(V_ISM[0], V_ISM[1] + 1e-9, V_SKEN_KORAK):
        if any(abs(v - c[0]) < 1.01 * sep for c in ism):
            continue
        nov = list(ism) + [(v, 2.0, 1.0, 0.5)]
        rss = float(np.sum((y - model.spekter(x, zv, nov)) ** 2))
        kandidati.append((rss, v))
    if not kandidati:
        return None
    kandidati.sort()

    izbrani = []
    for _, v in kandidati:
        if all(abs(v - u) >= 15.0 for u in izbrani):
            izbrani.append(v)
        if len(izbrani) >= N_ZACETKOV:
            break

    najboljsi = None
    for v in izbrani:
        nov = list(ism) + [(v, 2.0, 1.0, 0.5)]
        r = prilagodi(model, x, y, sestavi_p(p_zvezda, nov), len(nov), rv_sme)
        if najboljsi is None or r["rss"] < najboljsi["rss"]:
            najboljsi = r
    return najboljsi


def sigma_iz_diferenc(y):
    """Ocena šuma iz razlik sosednjih pikslov (robustna, MAD).

    Zakaj: spodnja meja za sigma pri BIC, če bi bil ostanek najkompleksnejšega
    modela nerealno majhen (npr. prevelik fit).

    Args:
        y: normaliziran spekter na večjem območju (brez izbire Na D).

    Returns:
        ocenjena sigma šuma enega piksla (v enotah normaliziranega fluksa).
    """
    d = np.diff(y)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def izberi_stevilo_komponent(model, x, y, rv_sme, sigma_mad):
    """Prilega modele z N_ISM = 1..N_ISM_MAX in po BIC izbere optimalno število.

    Zakaj: število ISM oblakov ni vnaprej znano; BIC kaznuje dodatne parametre
    (k * ln M). Dodatna komponenta je sprejeta le, če BIC pade za več kot
    DELTA_BIC glede na trenutni najboljši model; ob prvi zavrnitvi se iskanje
    ustavi (obdrži se enostavnejši model).

    Args:
        model: objekt ModelNaD.
        x: valovne dolžine pikslov okna fita [A].
        y: normaliziran opazovani fluks na x.
        rv_sme: RV zvezde iz SME [km/s] (začetni približek zvezdne hitrosti).
        sigma_mad: ocena šuma iz sosednjih pikslov (spodnja meja za sigma).

    Returns:
        slovar: fits (n -> rezultat fita, dopolnjen s k, chi2, bic), n_best
        (izbrano število ISM), sigma (uporabljena za chi2), M (št. pikslov).
    """
    M = len(x)
    p_zvezda = predfit_zvezde(model, x, y, rv_sme)
    fits, ism = {}, []
    for n in range(1, N_ISM_MAX + 1):
        r = dodaj_komponento(model, x, y, p_zvezda, ism, rv_sme)
        if r is None:
            break
        fits[n] = r
        p_zvezda = r["p"][:N_ZVEZDA]
        ism = razcleni_p(r["p"], n)[1]

    # sigma iz ostanka najkompleksnejšega modela, vsaj kot šum sosednjih pikslov
    n_max = max(fits)
    k_max = N_ZVEZDA + 4 * n_max
    sigma2 = max(fits[n_max]["rss"] / max(M - k_max, 1), sigma_mad ** 2)

    for n, r in fits.items():
        r["k"] = N_ZVEZDA + 4 * n
        r["chi2"] = r["rss"] / sigma2
        r["bic"] = r["chi2"] + r["k"] * np.log(M)

    n_best = 1
    for n in range(2, n_max + 1):
        if fits[n]["bic"] - fits[n_best]["bic"] < -DELTA_BIC:
            n_best = n
        else:
            break
    return dict(fits=fits, n_best=n_best, sigma=float(np.sqrt(sigma2)), M=M)


# ---------------------------------------------------------------
# NAPAKE IN KOMPONENTE
# ---------------------------------------------------------------
def napake_fizicnih_parametrov(r, M):
    """Napake fizikalnih parametrov iz kovariančne matrike fita.

    Zakaj: cov(p) = s^2 (J^T J)^-1, kjer je s^2 = RSS / (M - k) (ostanek tega
    modela); v fizikalne parametre se prenese prek Jacobijana preslikave.

    Args:
        r: rezultat fita (slovar iz prilagodi, z ključi p, rss, jac, n).
        M: število pikslov fita.

    Returns:
        polje napak (1 sigma) fizikalnih parametrov v vrstnem redu fizicni_vektor.
    """
    n = r["n"]
    k = N_ZVEZDA + 4 * n
    s2 = r["rss"] / max(M - k, 1)
    J = r["jac"][:M]
    cov_p = s2 * np.linalg.pinv(J.T @ J, rcond=1e-10)
    Jf = jakobijan_fizicni(r["p"], n)
    cov_f = Jf @ cov_p @ Jf.T
    return np.sqrt(np.clip(np.diag(cov_f), 0.0, None))


def parametri_na_mejah(r, rv_sme, dv_inst):
    """Imena notranjih parametrov, ki so na meji (napaka ni zanesljiva).

    Args:
        r: rezultat fita (slovar iz prilagodi).
        rv_sme: RV zvezde iz SME [km/s].
        dv_inst: instrumentalna ločljivost c/R [km/s].

    Returns:
        seznam imen parametrov znotraj 0.1 % razpona od spodnje ali zgornje meje.
    """
    n = r["n"]
    lo, hi, _ = meje_p(n, rv_sme, dv_inst)
    tol = 1e-3 * (hi - lo)
    imena = imena_p(n)
    return [imena[j] for j in range(len(imena))
            if r["p"][j] <= lo[j] + tol[j] or r["p"][j] >= hi[j] - tol[j]]


def sestavi_komponente(r, M, dv_inst):
    """Fizikalne komponente (zvezda + ISM) z napakami fita in celotnimi napakami.

    Zakaj: celotna napaka v in b vključuje sistematiko instrumenta (dv_inst/10,
    dv_inst/20), tau pri ISM sistematiko kontinuuma (0.02) - enako kot prej.

    Args:
        r: rezultat fita (slovar iz prilagodi).
        M: število pikslov fita.
        dv_inst: instrumentalna ločljivost c/R [km/s].

    Returns:
        seznam slovarjev: ime ('zvezda', 'ISM1', ...), v, b, tau_d2, tau_d1 (+ a pri
        zvezdi), vsak z e_* (napaka fita) in t_* (celotna napaka) za v, b, tau.
    """
    n = r["n"]
    f = fizicni_vektor(r["p"], n)
    e = napake_fizicnih_parametrov(r, M)
    sv, sb = dv_inst / SYS_V_DELITELJ, dv_inst / SYS_B_DELITELJ

    komp = [dict(
        ime="zvezda", v=f[0], e_v=e[0], t_v=np.hypot(e[0], sv),
        b=f[1], e_b=e[1], t_b=np.hypot(e[1], sb),
        a=f[2], e_a=e[2],
        tau_d2=f[3], e_tau_d2=e[3], t_tau_d2=e[3],
        tau_d1=f[4], e_tau_d1=e[4], t_tau_d1=e[4],
    )]
    for i in range(n):
        o = N_ZVEZDA + 4 * i
        komp.append(dict(
            ime=f"ISM{i + 1}", v=f[o], e_v=e[o], t_v=np.hypot(e[o], sv),
            b=f[o + 1], e_b=e[o + 1], t_b=np.hypot(e[o + 1], sb),
            a=np.nan, e_a=np.nan,
            tau_d2=f[o + 2], e_tau_d2=e[o + 2], t_tau_d2=np.hypot(e[o + 2], SYS_TAU),
            tau_d1=f[o + 3], e_tau_d1=e[o + 3], t_tau_d1=np.hypot(e[o + 3], SYS_TAU),
        ))
    return komp


# ---------------------------------------------------------------
# IZHODI
# ---------------------------------------------------------------
def narisi(pot_png, rez):
    """Nariše diagnostično sliko fita in jo shrani kot PNG.

    Zakaj: vizualna kontrola. Zgoraj opazovanje, skupni fit, zvezda in vsak ISM
    oblak posebej (vsak konvolviran z instrumentom) z izpisom v, b, tau in
    končnim N_ISM; spodaj ostanki.

    Args:
        pot_png: izhodna pot slike.
        rez: slovar rezultata spektra (ključi: objekt, instrument, spekter,
            x, y_norm, fit, f_zvezda, f_ism, komp, n_ism, bic_tab, sigma, rv_sme).
    """
    x = rez["x"]
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, gridspec_kw=dict(height_ratios=[3, 1])
    )
    barve = ["tab:green", "tab:purple", "tab:cyan", "tab:olive"]

    ax.plot(x, rez["y_norm"], color="black", lw=1.0, alpha=0.8, label="Opazovanje")
    ax.plot(x, rez["fit"], color="blue", lw=2.2, label=f"Skupni fit (N_ISM = {rez['n_ism']})")
    ax.plot(x, rez["f_zvezda"], color="tab:red", lw=1.5, ls="--", label="Zvezda")
    for i, f in enumerate(rez["f_ism"]):
        ax.plot(x, f, color=barve[i % len(barve)], lw=1.5, ls="--", label=f"ISM {i + 1}")
    # Pozicije ISM dubleta (D2 in D1)
    for i, c in enumerate(rez["komp"][1:]):
        for lam0 in (W_D2, W_D1):
            ax.axvline(lam0 * (1 + c["v"] / C_KMS), color=barve[i % len(barve)],
                       lw=0.8, ls=":")

    c = rez["komp"][0]
    vrste = [
        f"Zvezda: v = {c['v']:.1f} ± {c['t_v']:.1f} km/s,  b = {c['b']:.1f} ± {c['t_b']:.1f} km/s,"
        f"  a = {c['a']:.2f}",
        f"        τD2 = {c['tau_d2']:.1f},  τD1 = {c['tau_d1']:.1f}   (rv_sme = {rez['rv_sme']:.1f} km/s)",
    ]
    for c in rez["komp"][1:]:
        vrste.append(
            f"{c['ime']}: v = {c['v']:.1f} ± {c['t_v']:.1f} km/s,  b = {c['b']:.2f} ± {c['t_b']:.2f} km/s,"
            f"  τD2 = {c['tau_d2']:.2f} ± {c['t_tau_d2']:.2f},  τD1 = {c['tau_d1']:.2f} ± {c['t_tau_d1']:.2f}"
        )
    vrste.append(f"N_ISM = {rez['n_ism']}")
    ax.text(0.015, 0.03, "\n".join(vrste), transform=ax.transAxes, fontsize=9,
            va="bottom", bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))

    bic = "   ".join(f"N={t['n']}: BIC={t['bic']:.0f} (Δ={t['dbic']:+.0f})" for t in rez["bic_tab"])
    ax.set_title(f"{rez['objekt']} ({rez['instrument']}) - {rez['spekter']}\n{bic}", fontsize=11)
    ax.set_ylabel("Normaliziran fluks")
    ax.set_ylim(min(0.0, np.nanmin(rez["y_norm"])) - 0.05, 1.3)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)

    axr.plot(x, rez["y_norm"] - rez["fit"], color="black", lw=0.9)
    axr.axhline(0, color="gray")
    axr.axhline(rez["sigma"], color="gray", ls=":")
    axr.axhline(-rez["sigma"], color="gray", ls=":")
    axr.set_ylabel("Opaz. - fit")
    axr.set_xlabel("Valovna dolžina [Å]")
    axr.set_xlim(*FIT_NA)
    axr.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(pot_png, dpi=100)
    plt.close()


def zapisi_data(pot_txt, rez):
    """Zapiše podrobne rezultate enega spektra v besedilno datoteko.

    Zakaj: preglednost in nadaljnja analiza; poleg parametrov z napakami vsebuje
    tabelo BIC in stolpce spektra, fita in posameznih komponent.

    Args:
        pot_txt: izhodna pot datoteke (_data.txt).
        rez: slovar rezultata spektra (glej obdelaj_spekter).
    """
    with open(pot_txt, "w") as f:
        f.write(f"# File: {rez['fits_pot']}\n")
        f.write(f"# Resolution R: {rez['R']}  (dv_inst = {rez['dv_inst']:.2f} km/s)\n")
        f.write(f"# rv_sme = {rez['rv_sme']:.3f} km/s, b_offset = {rez['b_offset']:+.4f}, "
                f"c_slope = {rez['c_slope']:+.6f}\n")
        f.write(f"# Fit okno: {FIT_NA[0]}-{FIT_NA[1]} A, M = {rez['M']} pikslov, "
                f"sigma (za chi2) = {rez['sigma']:.4f}\n")
        f.write(f"# Sistematika: err_v = {rez['dv_inst'] / SYS_V_DELITELJ:.3f} km/s, "
                f"err_b = {rez['dv_inst'] / SYS_B_DELITELJ:.3f} km/s, err_tau(ISM) = {SYS_TAU}\n")
        f.write("# ---------------------------------------------------------\n")
        f.write("# BIC po številu ISM komponent (dBIC glede na N-1):\n")
        for t in rez["bic_tab"]:
            f.write(f"#   N_ISM={t['n']}  k={t['k']:2d}  RSS={t['rss']:.5f}  chi2={t['chi2']:.1f}  "
                    f"BIC={t['bic']:.1f}  dBIC={t['dbic']:+.1f}\n")
        f.write(f"# IZBRANO: N_ISM = {rez['n_ism']}\n")
        f.write("# ---------------------------------------------------------\n")
        f.write("# Parametri (vrednost +- napaka fita | celotna napaka):\n")
        for c in rez["komp"]:
            f.write(f"# {c['ime']}:\n")
            f.write(f"#   v_rad = {c['v']:8.3f} +- {c['e_v']:6.3f} | Tot: {c['t_v']:6.3f} km/s\n")
            f.write(f"#   b     = {c['b']:8.3f} +- {c['e_b']:6.3f} | Tot: {c['t_b']:6.3f} km/s\n")
            if c["ime"] == "zvezda":
                f.write(f"#   a     = {c['a']:8.3f} +- {c['e_a']:6.3f}   (dusenje Voigta)\n")
            f.write(f"#   tD2   = {c['tau_d2']:8.3f} +- {c['e_tau_d2']:6.3f} | Tot: {c['t_tau_d2']:6.3f}\n")
            f.write(f"#   tD1   = {c['tau_d1']:8.3f} +- {c['e_tau_d1']:6.3f} | Tot: {c['t_tau_d1']:6.3f}\n")
        if rez["na_meji"]:
            f.write(f"# OPOZORILO: parametri na meji (napaka ni zanesljiva): {', '.join(rez['na_meji'])}\n")
        f.write("# ---------------------------------------------------------\n")
        glava = "# Wave, Flux_raw, Flux_norm, Fit, Zvezda" + "".join(
            f", ISM_{i + 1}" for i in range(rez["n_ism"]))
        f.write(glava + "\n")
        for i in range(len(rez["x"])):
            vrstica = [rez["x"][i], rez["y_raw"][i], rez["y_norm"][i], rez["fit"][i], rez["f_zvezda"][i]]
            vrstica += [fi[i] for fi in rez["f_ism"]]
            f.write(f"{vrstica[0]:.4f} " + " ".join(f"{v:.4f}" for v in vrstica[1:]) + "\n")


def zapisi_tabelo(vrstice, pot):
    """Zapiše skupno tabelo vseh komponent vseh spektrov.

    Zakaj: pregleden pregled rezultatov za vse spektre (npr. porazdelitev ISM
    hitrosti po objektih) v enem mestu.

    Args:
        vrstice: seznam slovarjev (ena komponenta = ena vrstica), ključi kot v
            obdelaj_spekter.
        pot: izhodna pot (TABELA_TXT).
    """
    vrstice = sorted(vrstice, key=lambda r: (r["objekt"], r["spekter"], r["komp"]))
    with open(pot, "w") as f:
        f.write("# Na D: zvezda + ISM (Voigt). Napake: e = fit, t = celotna (s sistematiko).\n")
        f.write("# N_ISM po BIC; chi2_red glede na skupno sigma spektra.\n")
        f.write("objekt instrument spekter N_ISM komp v e_v t_v b e_b t_b "
                "tau_D2 e_tau_D2 t_tau_D2 tau_D1 e_tau_D1 t_tau_D1 a_damp chi2_red\n")
        for r in vrstice:
            f.write(
                f"{r['objekt']} {r['instrument']} {r['spekter']} {r['n_ism']} {r['komp']} "
                f"{r['v']:.3f} {r['e_v']:.3f} {r['t_v']:.3f} "
                f"{r['b']:.3f} {r['e_b']:.3f} {r['t_b']:.3f} "
                f"{r['tau_d2']:.3f} {r['e_tau_d2']:.3f} {r['t_tau_d2']:.3f} "
                f"{r['tau_d1']:.3f} {r['e_tau_d1']:.3f} {r['t_tau_d1']:.3f} "
                f"{r['a']:.3f} {r['chi2_red']:.2f}\n"
            )


# ---------------------------------------------------------------
# OBDELAVA SPEKTRA
# ---------------------------------------------------------------
def _obdelaj(vrstica, fits_pot):
    """Celotna obdelava enega spektra: normalizacija, fit, izbira N_ISM, izhodi.

    Args:
        vrstica: slovar vrstice tabele rv_sme_rezultati.txt (objekt, instrument,
            spekter, rv_sme, b_offset, c_slope, ...).
        fits_pot: pot do FITS datoteke tega spektra.

    Returns:
        slovar: ok (True), povzetek (niz za izpis) in vrstice (seznam slovarjev
        komponent za skupno tabelo).
    """
    objekt, instrument = str(vrstica["objekt"]).upper(), str(vrstica["instrument"]).lower()
    ime = str(vrstica["spekter"])
    rv_sme = float(vrstica["rv_sme"])
    b_off, c_slope = float(vrstica["b_offset"]), float(vrstica["c_slope"])

    # Surov spekter -> normalizacija s b in c (kot normalizacija.py)
    wave, flux = preberi_spekter(fits_pot)
    x_vse, y_vse, _, x_sred = izrezi_okno(wave, flux)
    y_norm_vse, _ = normaliziraj_bc(x_vse, y_vse, x_sred, b_off, c_slope)
    sigma_mad = sigma_iz_diferenc(y_norm_vse)

    m = (x_vse >= FIT_NA[0]) & (x_vse <= FIT_NA[1])
    x, y_raw, y = x_vse[m], y_vse[m], y_norm_vse[m]

    R = RESOLUCIJA.get(instrument, RESOLUCIJA_DEFAULT)
    model = ModelNaD(FIT_NA[0], FIT_NA[1], R)

    izb = izberi_stevilo_komponent(model, x, y, rv_sme, sigma_mad)
    fits, n_best, M = izb["fits"], izb["n_best"], izb["M"]
    r = fits[n_best]

    # BIC tabela (dBIC glede na prejšnji N)
    bic_tab = []
    for n in sorted(fits):
        t = fits[n]
        dbic = t["bic"] - fits[n - 1]["bic"] if n > 1 else 0.0
        bic_tab.append(dict(n=n, k=t["k"], rss=t["rss"], chi2=t["chi2"], bic=t["bic"], dbic=dbic))

    # Komponente z napakami in krivulje za sliko
    komp = sestavi_komponente(r, M, model.dv_inst)
    zv, ism = razcleni_p(r["p"], n_best)
    fit = model.spekter(x, zv, ism)
    f_zv = model.spekter(x, zv, [])
    f_ism = [model.spekter(x, None, [c]) for c in ism]

    k = N_ZVEZDA + 4 * n_best
    chi2_red = r["chi2"] / max(M - k, 1)
    rez = dict(
        fits_pot=fits_pot, objekt=objekt, instrument=instrument, spekter=ime,
        R=R, dv_inst=model.dv_inst, rv_sme=rv_sme, b_offset=b_off, c_slope=c_slope,
        M=M, sigma=izb["sigma"], bic_tab=bic_tab, n_ism=n_best, komp=komp,
        na_meji=parametri_na_mejah(r, rv_sme, model.dv_inst),
        x=x, y_raw=y_raw, y_norm=y, fit=fit, f_zvezda=f_zv, f_ism=f_ism,
    )

    rel = os.path.relpath(fits_pot, start=SPEKTRI_KOREN)
    izhod_dir = os.path.join(IZHOD_DIR, os.path.dirname(rel))
    os.makedirs(izhod_dir, exist_ok=True)
    osnova = os.path.join(izhod_dir, ime.replace(".fits", ""))
    zapisi_data(osnova + "_data.txt", rez)
    narisi(osnova + "_fit.png", rez)

    vrstice = []
    for c in komp:
        vrstice.append(dict(
            objekt=objekt, instrument=instrument, spekter=ime, n_ism=n_best,
            komp=c["ime"], v=c["v"], e_v=c["e_v"], t_v=c["t_v"],
            b=c["b"], e_b=c["e_b"], t_b=c["t_b"],
            tau_d2=c["tau_d2"], e_tau_d2=c["e_tau_d2"], t_tau_d2=c["t_tau_d2"],
            tau_d1=c["tau_d1"], e_tau_d1=c["e_tau_d1"], t_tau_d1=c["t_tau_d1"],
            a=c["a"], chi2_red=chi2_red,
        ))
    ism_v = ", ".join(f"{c['v']:+.1f}" for c in komp[1:])
    povzetek = (f"{objekt} {ime}: N_ISM={n_best}, v_zv={komp[0]['v']:+.1f} "
                f"(rv_sme {rv_sme:+.1f}), v_ISM=[{ism_v}], chi2_red={chi2_red:.2f}")
    return dict(ok=True, povzetek=povzetek, vrstice=vrstice)


def obdelaj_spekter(naloga):
    """Ovojnica okoli _obdelaj, ki ujame napake (za vzporedno obdelavo).

    Zakaj: napaka pri enem spektru ne sme ustaviti obdelave ostalih; funkcija je
    na vrhnjem nivoju modula, da jo multiprocessing lahko pošlje procesom.

    Args:
        naloga: par (vrstica, fits_pot) - slovar vrstice tabele in pot do FITS.

    Returns:
        rezultat _obdelaj, ali slovar (ok=False, povzetek=opis napake).
    """
    vrstica, fits_pot = naloga
    try:
        return _obdelaj(vrstica, fits_pot)
    except Exception as e:
        return dict(ok=False, vrstice=[],
                    povzetek=f"{vrstica['objekt']} {vrstica['spekter']}: NAPAKA: {e}")


def main(filter_objekti=None):
    """Obdela vse (ali izbrane) spektre iz tabele in zapiše skupno tabelo.

    Zakaj: vstopna točka. Spektre obdela vzporedno (N_PROCESOV), skupno tabelo
    pa sproti prepiše, da se ob prekinitvi rezultati ne izgubijo.

    Args:
        filter_objekti: seznam imen objektov (velike črke, npr. ["A2-342"]);
            če None, se obdelajo vsi objekti iz tabele.
    """
    if not os.path.exists(REZULTATI_TXT):
        print(f"Napaka: {REZULTATI_TXT} ne obstaja. Najprej zaženite sme_rv_vsi.py!")
        return
    os.makedirs(IZHOD_DIR, exist_ok=True)

    df = pd.read_csv(REZULTATI_TXT, sep=r"\s+", comment="#")
    if filter_objekti:
        df = df[df["objekt"].str.upper().isin(filter_objekti)]

    # Indeks FITS datotek (objekt, instrument, ime) -> pot
    poti = {}
    for p in glob.glob(SPEKTRI_GLOB):
        deli = p.replace("\\", "/").split("/")
        poti[(deli[-3].upper(), deli[-2].lower(), deli[-1])] = p

    naloge = []
    for _, v in df.iterrows():
        pot = poti.get((str(v["objekt"]).upper(), str(v["instrument"]).lower(), str(v["spekter"])))
        if pot is None:
            print(f"[Opozorilo] {v['spekter']} ni na disku. Preskakujem.")
            continue
        naloge.append((dict(v), pot))
    print(f"Spektrov za obdelavo: {len(naloge)} (procesov: {N_PROCESOV})")

    vrstice, napake = [], 0

    def sprejmi(i, res):
        nonlocal napake
        print(f"[{i}/{len(naloge)}] {res['povzetek']}")
        if res["ok"]:
            vrstice.extend(res["vrstice"])
        else:
            napake += 1
        if i % 10 == 0:
            zapisi_tabelo(vrstice, TABELA_TXT)

    if N_PROCESOV > 1 and len(naloge) > 1:
        with mp.Pool(N_PROCESOV) as pool:
            for i, res in enumerate(pool.imap_unordered(obdelaj_spekter, naloge), 1):
                sprejmi(i, res)
    else:
        for i, nal in enumerate(naloge, 1):
            sprejmi(i, obdelaj_spekter(nal))

    zapisi_tabelo(vrstice, TABELA_TXT)
    print(f"\nGotovo! Napak: {napake}. Slike in _data.txt: {IZHOD_DIR}/<objekt>/<instrument>/, "
          f"tabela: {TABELA_TXT}")


if __name__ == "__main__":
    main([a.upper() for a in sys.argv[1:]] or None)
