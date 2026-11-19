# HANDOVER – SME RV fit za M-gigante (magistrska)

Datum: 2026-11-02. Delovna mapa: `/home/gueswho/Documents/Faks/Magistrska`
Jezik komunikacije z uporabnikom: slovenščina. Uporabnik zažene kodo SAM (agent kode ne zaganja, razen če uporabnik izrecno reče).

## 1. Glavni cilj
Magistrska naloga: analiza spektrov (Asiago/Varese ešele) 42 M-gigantov (A2-xxx), območje Na D (5845–5945 Å).
Trenutna podnaloga: **določiti radialno hitrost (RV) iz SME sintetičnih spektrov**, brez `pysme.solve`
(prepočasen). Način: model se sintetizira enkrat (mreža teff × monh × dNa, predpomnjeno v pickle),
nato se samo premika po RV (`np.interp`) in išče minimum χ². Začetni RV iz literature (GALAH DR3, sicer Gaia DR3,
`zvezde_RV_iz_literature.csv`). Nato: boljša normalizacija kontinuuma (polinom skozi točke, kjer je SME model
na kontinuumu, npr. > 0.99).

## 2. Doseženo
- Zbrane literaturne RV (Gaia+GALAH) preko `rv_survey.py`; popravljeno branje stolpcev GALAH (`RVgalah`).
- `sme_rv_vsi.py` (trenutna verzija, uporabnik jo je močno predelal):
  - 2-stopenjski fit: grobi RV (±40 km/s, z Na D) → fini RV (±6 km/s, maskiran Na D, samo kovinske črte).
  - Aditivni offset `b` + naklon `c` (brez množenja `a`), shranjena v rezultatih za kasnejšo normalizacijo.
  - Model objekta izbran iz mreže (13 Teff × 6 monh × 6 dNa) na N_IZBOR spektrih (log χ²).
  - Maska Na D zožana (zvezdni 5885–5900 Å, ISM 5888.5–5897.5 Å); nastavljiva z `MASKA_ZVEZDNI`, `MASKA_ISM`.
- Starejša verzija `sme_rv_vsi_old.py` (množilni `a`, 1 prehod + široki ±100 km/s prehod, zastavice `na_robu`,
  `odstopa`); njeni rezultati so bili analizirani, napisane so bile opombe za sestanek s profesorjem.
- Analiza rezultatov: 310 spektrov / 41 objektov; sistematičen negativen odmik SME−lit ~ −5…−9 km/s pri več zvezdah.

## 3. Ključne datoteke
- `sme_rv_vsi.py` – trenutni skript (2-stopenjski RV, aditivni b/c, maska Na D). Zadnja sprememba: zožana maska.
- `sme_rv_vsi_old.py` – stara verzija (množilni a, široki prehod, `odstopa`).
- `rv_survey.py` – crossmatch A2_list_all.txt z Gaia DR3 / GALAH DR3 (astroquery XMatch).
- `zvezde_RV_iz_literature.csv` – literaturni RV (stolpca `RV_Gaia [km/s]`, `RV_GALAH [km/s]`).
- `opombe_cudni_spektri.txt` – opombe za profesorja (odstopajoči spektri); **izhaja iz STARE kode**, ni več aktualna za novo kodo.
- `normalizacija.py`, `opazke.txt`, `test.ipynb` – uporabnikove datoteke (nisem jih spreminjal).
- Izhodi: `figs/sme_rv/rv_sme_rezultati.txt` (stolpci: objekt, instrument, spekter, rv_lit, vir, rv_sme, e_rv_sme,
  razlika, b_offset, c_slope, teff, monh, dna, sigma, na_robu), `figs/sme_rv/rv_sme_modeli_objektov.txt`,
  `figs/sme_rv/modeli_cache.pkl` (predpomnjeni SME modeli – NE brisati, sinteza je draga), slike fitov v `figs/sme_rv/`.
- `VALD_PATH = vald_3800K_5800_6000` – line list.

## 4. Trenutno stanje sistema
- Git: veja `main`; večina datotek je neprijavljenih (untracked, npr. `A2_list_all.txt`, `Paper_II.tar.gz`,
  `Rezultati/`). Nič ni bilo commitano po moji strani.
- Spremembe maske v `sme_rv_vsi.py` **še niso bile zagnane**; rezultati v `figs/sme_rv/` so iz prejšnjega zagona
  (stara maska 5882–5903 Å).
- Ni zagnanih procesov v ozadju. Ni testov.

## 5. Blokade in odprta vprašanja
- **Problem:** pri objektih A2-091, 184, 224, 234, 235, 236, 342, 354 izbere izbira modela hladne (Teff 3300–3600 K),
  metalno bogate modele, ki se ne prilegajo. Vzrok (iz rezultatov): mediana `b_offset` ≈ 0.6–0.85 (pri ostalih ≈ 0.09),
  torej model ima povprečje ~0.13 in ga offset dvigne; sigma je pri obeh skupinah enaka; krivulja χ² je ravna.
  Fit izbere skoraj ravno črto (povprečje) – RV ni določen, e_rv ±1.5 je podcenjena.
- Dodatno: izbira modela uporablja RV iz enega spektra s privzetim modelom (3500/0/0) in okno ±4 km/s ter NE
  maskira Na D (drugače kot končni fit).
- Uporabnik NE želi: razširitve mreže (predolga sinteza), prostega množilnega `a` kot "umetnega podaljševanja črt".
- Čaka potrditev (moji predlogi): (1) omejitev |b| < ~0.2 (ali kazen) pri izbiri modela; (2) zastavica
  "RV ni določen" ko je krivulja χ² ravna; (3) oznaka "ni zanesljivega modela" ali model podobne zvezde.
- Odprto: ali so spektri heliocentrično popravljeni (predpostavljeno da).
- A2-028/20231028_ech_a2-023_hv_c.fits verjetno napačno razvrščen (ime pravi a2-023).
- Normalizacija po SME (polinom skozi točke, kjer je model > 0.99, sigma-clipping, iteracija RV↔kontinuum)
  je dogovorjena na konceptualni ravni, koda še ni napisana.

## 6. Naslednji 3 koraki
1. Implementirati v `sme_rv_vsi.py` omejitev `b` (npr. |b| < 0.2 ali kazen (b/0.1)²) pri izbiri modela in
   izbiro modela narediti dosledno z končnim fitom (RV na spekter, maska Na D); `b` za končni RV pustiti prost.
2. Dodati zastavico "RV ni določen" (relativna razlika χ² čez krivuljo RV pod pragom) in po zagonu preveriti
   objekte 091, 184, 224, 234, 235, 236, 342, 354 (izbrani Teff/monh, `b_offset`).
3. Napisati normalizacijo kontinuuma (korak 6): iz najboljšega modela vzeti točke z fluksom > ~0.97–0.99, na
   opazovanih spektrih fittati polinom (st. 1–2, sigma-clipping), iterirati RV↔kontinuum 2–3 krogi; primerjati
   RV pri različnih stopnjah polinoma. Nato posodobiti `opombe_cudni_spektri.txt` z novimi rezultati.
