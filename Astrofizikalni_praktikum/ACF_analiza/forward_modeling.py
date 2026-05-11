import pandas as pd
import numpy as np
import datetime
import matplotlib
matplotlib.use('Agg') # Uporabi neinteraktivni backend za shranjevanje brez displaya
import matplotlib.pyplot as plt
from astropy.time import Time
import matplotlib.dates as mdates
import scipy.signal as signal

# Tukaj vpiši stevilo vseh zvezd katerim spremljamo fotometrijo
stevilo_vseh_zvezd = 2

# 1. Naložimo podatke
# Uporabimo regularni izraz '\s+' za ločevanje stolpcev (več presledkov)
stolpci = ['IMAGE', 'XCENTER', 'YCENTER', 'MAG', 'MERR', 'IFILTER', 'OTIME', 'ID']
pot_do_podatkov = 'rezultati_fotometrija_AG_Dra.txt'

df = pd.read_csv(pot_do_podatkov, sep=r'\s+', names=stolpci)

# 2. Očistimo 'INDEF' vrednosti in pretvorimo v številke
# IRAF zapiše INDEF, kadar zvezde ne more izmeriti. To zamenjamo z NaN in odstranimo.
df['MAG'] = pd.to_numeric(df['MAG'], errors='coerce')
df['MERR'] = pd.to_numeric(df['MERR'], errors='coerce')
df['OTIME'] = pd.to_numeric(df['OTIME'], errors='coerce')
df['XCENTER'] = pd.to_numeric(df['XCENTER'], errors='coerce')
df['YCENTER'] = pd.to_numeric(df['YCENTER'], errors='coerce')

# Določitev pravega ID-ja pred kakršnimkoli brisanjem!
# Uporabimo matematični trik, o katerem sva prej govorila
df['PRAVI_ID'] = (df['ID'] - 1) % stevilo_vseh_zvezd + 1
# Sedaj lahko pobrišemo INDEF meritve
df = df.dropna(subset=['MAG', 'OTIME'])

# TUKAJ VPIŠI PRAVE ID-JE:
mira_id = 1          # ID Mire v big anulus podatkih je 2

ref_zvezda_id = 2



# 1. Izluščimo samo našo referenčno zvezdo in obdržimo LE nujne stolpce
ref_podatki = df[df['PRAVI_ID'] == ref_zvezda_id][['IMAGE', 'MAG', 'MERR']]

# 2. Preimenujemo no njena stolpca, da ju kasneje ločimo od Mire
ref_podatki = ref_podatki.rename(columns={'MAG': 'REF_MAG', 'MERR': 'REF_MERR'})

# 3. Združimo to referenčno "bazo" nazaj z glavno tabelo (povežemo preko pripadajoče slike)
df = pd.merge(df, ref_podatki, on='IMAGE')

# 4. Izračunamo razliko (diferencialno magnitudo) + dodamo tvojo konstanto 14.52
df['DIFF_MAG'] = df['MAG'] - df['REF_MAG'] + 14.52

# 5. Nova napaka je koren vsote kvadratov obeh posameznih napak
df['DIFF_MERR'] = np.sqrt(df['MERR']**2 + df['REF_MERR']**2)



# --------- ZDAJ LAHKO IZLUŠČIMO SAMO MIRO ---------
mira_df = df[df['PRAVI_ID'] == mira_id].copy()
# Pretvorba MJD v prave ure (datetime format)
casi = Time(mira_df['OTIME'].values, format='mjd')

# Lahko dodaš še 1 uro (ali 2 poleti) za slovenski lokalni čas, če želiš:
# from datetime import timedelta
# mira_df['URE'] = casi.datetime + timedelta(hours=1)

mira_df['URE'] = casi.datetime

# Sortiramo da bo krivulja pravilna
mira_df = mira_df.sort_values(by='URE')

# Tu odščipni morebitne divje podatke (recimo odreži prvih 5 in zadnjih 50 slik)
mira_df = mira_df[:-110]


# 1. Priprava podatkov
# Vzamemo samo podatke, kjer rolling average ni NaN
valid_data = mira_df.dropna(subset=['DIFF_MAG']).copy()

# Čas pretvorimo v sekunde od začetka opazovanja
t_sec = (valid_data['URE'] - valid_data['URE'].iloc[0]).dt.total_seconds().values
mag = valid_data['DIFF_MAG'].values

# Za ACF moramo signal centrirati okoli 0 (odštejemo povprečje in trend ki je linearen)
mag_anom = signal.detrend(mag, type='linear')


# odstranitev kvadratnega trenda (uporaba numpy)
koeficienti = np.polyfit(t_sec, mag, 2)     # Poiščemo najboljšo parabolo (polinom 2. stopnje)
trend = np.polyval(koeficienti, t_sec)      # Izračunamo vrednosti parabole za vse čase
mag_anom = mag - trend                      # Odštejemo parabolo od originalnega signala

# 2. Izračun Avtokorelacijske funkcije (ACF)
# Uporabimo numpy.correlate. 'full' vrne korelacije za vse možne zamike
acf_full = np.correlate(mag_anom, mag_anom, mode='full')

# Zanimajo nas samo pozitivni zamiki (druga polovica niza)
acf = acf_full[len(acf_full)//2:]

# Normiramo ACF tako, da je maksimalna korelacija (pri zamiku 0) enaka 1
acf = acf / acf[0]

# Določimo časovne zamike (lags) v sekundah in minutah
# Predpostavimo povprečen razmik med posnetki (ekspozicija + readout)
dt_mean = np.median(np.diff(t_sec))
lags_sec = np.arange(len(acf)) * dt_mean
lags_min = lags_sec / 60.0


# ---------------------------------------------------------
# 1. FUNKCIJA ZA HITRO SIMULACIJO IN IZRAČUN ACF
# ---------------------------------------------------------
def simuliraj_in_izracunaj_acf(tau_min, f_sig, dt_min, max_lag_idx):
    """
    Ustvari dolg sintetični signal, da dobimo gladko statistiko,
    in izračuna njegovo ACF.
    f_sig: delež variance, ki je pravi signal (med 0 in 1)
    """
    N = 15000  # Daljši niz za gladko sintetično ACF
    y = np.zeros(N)
    
    tau_sec = tau_min * 60.0
    dt_sec = dt_min * 60.0
    
    decay = np.exp(-dt_sec / tau_sec)
    
    # Nastavimo skupno varianco na 1.0 (sigma_sig^2 = f_sig)
    sigma_sig = np.sqrt(f_sig)
    driving_std = np.sqrt((sigma_sig**2) * (1 - decay**2))
    
    # Hitra generacija zanke
    noise = np.random.normal(0, driving_std, N)
    for i in range(1, N):
        y[i] = y[i-1] * decay + noise[i]
        
    # Dodamo instrumentalni šum (sigma_noise^2 = 1 - f_sig)
    sigma_noise = np.sqrt(1.0 - f_sig)
    y_obs = y + np.random.normal(0, sigma_noise, N)
    
    # Izračunamo ACF
    y_anom = y_obs - np.mean(y_obs)
    acf_full = np.correlate(y_anom, y_anom, mode='full')
    acf_half = acf_full[N-1:]
    acf_norm = acf_half / acf_half[0]
    
    return acf_norm[:max_lag_idx]

# ---------------------------------------------------------
# 2. PRIPRAVA ZA FORWARD MODELING (MREŽA PARAMETROV)
# ---------------------------------------------------------
# PREDPOSTAVKA: Tvoja originalna ACF ('acf') in ('lags_min') sta že v spominu.
# Tukaj bomo vzeli dolžino tvoje opazovane ACF
max_idx = len(acf)
dt_mean_min = np.median(np.diff(lags_min)) # povprečen razmik med točkami v minutah

# Določimo meje faznega prostora (Grid Search)
tau_values = np.linspace(10, 60, 100)      # Preizkusimo tau od 10 do 60 minut (30 korakov) pri Miri je nekje 21min AG Dra pa 2min
f_sig_values = np.linspace(0.15, 0.6, 100) # Delež pravega signala od 50% do 99% pri miri je nekje 15$ ish sum pri Ag Dra pa 50%

# Matrika za shranjevanje napak (Sum of Squared Errors - SSE)
sse_matrix = np.zeros((len(f_sig_values), len(tau_values)))

print("Računam sintetične modele... To lahko traja nekaj sekund.")

best_sse = np.inf
best_tau = 0
best_fsig = 0
best_acf_synth = None

# Omejimo izračun SSE na časovne zamike do 100 minut
valid_lags = lags_min <= 50

total_iterations = len(f_sig_values) * len(tau_values)
start_time = datetime.datetime.now()
print(f"Začetek računanja: {start_time.strftime('%H:%M:%S')}")

iteration_count = 0

# Zanka čez vse kombinacije
for i, f_sig in enumerate(f_sig_values):
    for j, tau in enumerate(tau_values):
        # 1. Ustvarimo sintetično ACF
        acf_synth = simuliraj_in_izracunaj_acf(tau, f_sig, dt_mean_min, max_idx)
        
        # 2. Izračunamo razliko med opazovano in sintetično (SSE)
        # Računamo razliko samo za zamike do 100 minut!
        sse = np.sum((acf[valid_lags] - acf_synth[valid_lags])**2)
        sse_matrix[i, j] = sse
        
        # 3. Shranimo zmagovalca
        if sse < best_sse:
            best_sse = sse
            best_tau = tau
            best_fsig = f_sig
            best_acf_synth = acf_synth
            
        iteration_count += 1
        # Izpišemo vsak 1% napredka (ali vsaj na vsako iteracijo, če jih je zelo malo)
        if iteration_count % max(1, total_iterations // 100) == 0 or iteration_count == total_iterations:
            print(f"\rNapredek: {iteration_count/total_iterations * 100:.1f} %", end="", flush=True)

end_time = datetime.datetime.now()
print(f"\nKonec računanja: {end_time.strftime('%H:%M:%S')}")
print(f"Čas trajanja: {end_time - start_time}")

print(f"--- NAJBOLJŠI PARAMETRI (ZMAGOVALEC) ---")
print(f"Karakteristični čas (tau): {best_tau:.1f} minut")
print(f"Delež pravega signala: {best_fsig*100:.1f} %")
print(f"Delež šuma: {(1-best_fsig)*100:.1f} %")

# ---------------------------------------------------------
# 3. SHRANJEVANJE ZMAGOVALNEGA MODELA V .TXT DATOTEKO
# ---------------------------------------------------------
# Sedaj ustvarimo sintetično krivuljo točno iste dolžine kot so tvoji podatki
N_obs = len(mira_df)
y_best = np.zeros(N_obs)
decay_best = np.exp(-(dt_mean_min*60.0) / (best_tau*60.0))
driving_std_best = np.sqrt((best_fsig) * (1 - decay_best**2))

noise_sig = np.random.normal(0, driving_std_best, N_obs)
for i in range(1, N_obs):
    y_best[i] = y_best[i-1] * decay_best + noise_sig[i]

# Dodamo šum in postavimo na isti nivo kot tvoja povprečna magnituda
y_best_obs = y_best + np.random.normal(0, np.sqrt(1.0 - best_fsig), N_obs)
povprecna_mag = np.mean(mira_df['DIFF_MAG'])
amplituda_skaliranje = np.std(mira_df['DIFF_MAG']) / np.std(y_best_obs)
y_zmagovalec_koncni = povprecna_mag + (y_best_obs * amplituda_skaliranje)

# Shranimo v tekstovno datoteko
np.savetxt("Sinteticna_Zvezda_Zmagovalec.txt", 
           np.column_stack((mira_df['URE'].dt.strftime('%H:%M:%S').values, y_zmagovalec_koncni)), 
           fmt="%s", header="Čas\tSintetična_Magnituda")
print("Sintetična svetlobna krivulja uspešno shranjena v 'Sinteticna_Zvezda_Zmagovalec.txt'!")

# ---------------------------------------------------------
# 4. IZRIS FAZNEGA PROSTORA IN PRIMERJAVE ACF
# ---------------------------------------------------------
# PLOT 1: Fazni prostor (Contour map)
plt.figure(figsize=(8, 6))
X, Y = np.meshgrid(tau_values, f_sig_values)
# Uporabimo pcolormesh namesto contourf za 'blocky' videz (brez glajenja)
cp = plt.pcolormesh(X, Y, np.log10(sse_matrix), cmap='viridis', shading='auto')
plt.colorbar(cp, label='Log(Difference between model and observation)')
#plt.plot(best_tau, best_fsig, 'r*', markersize=15, label='Best fit')
plt.xlabel('Characteristic time $\\tau$ (minutes)')
plt.ylabel('True signal fraction ($f_{sig}$)')
plt.title('Forward Model Phase Space')
#plt.legend()
plt.tight_layout()
plt.savefig('figs/ag_dra_forward_model_fazni_prostor_bolj_natancno.png')
plt.close()

# PLOT 2: ACF Comparison (Observed vs Best Synthetic)
plt.figure(figsize=(8, 6))
plt.plot(lags_min, acf, 'k-', linewidth=2, label='Observed ACF')
plt.plot(lags_min, best_acf_synth, 'r--', linewidth=2,
         label=f'Best Synthetic ACF\n($\\tau$={best_tau:.1f}m, Noise={(1-best_fsig)*100:.0f}%)')
plt.axhline(0, color='gray', linestyle=':')
plt.xlabel('Time lag (minutes)')
plt.ylabel('Autocorrelation')
plt.title('Comparison of Observed and Synthetic ACF')
plt.legend()

plt.tight_layout()
plt.savefig('figs/ag_dra_forward_model_acf_primerjava_bolj_natancno.png')
plt.close()
