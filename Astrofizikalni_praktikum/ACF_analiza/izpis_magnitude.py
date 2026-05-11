import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.time import Time
# 1. Naložimo podatke

stolpci =['IMAGE', 'XCENTER', 'YCENTER', 'MAG', 'MERR', 'IFILTER', 'OTIME', 'ID']
pot_do_podatkov = 'rezultati_fotometrija_AG_Dra.txt'
df = pd.read_csv(pot_do_podatkov, sep=r'\s+', names=stolpci)

# 2. Očistimo 'INDEF' vrednosti (pretvorba v številke in brisanje NaN)
df['MAG'] = pd.to_numeric(df['MAG'], errors='coerce')
df['MERR'] = pd.to_numeric(df['MERR'], errors='coerce')
df['OTIME'] = pd.to_numeric(df['OTIME'], errors='coerce')
df = df.dropna(subset=['MAG', 'OTIME'])

# 3. Določimo prave ID-je
stevilo_vseh_zvezd = 2
df['PRAVI_ID'] = (df['ID'] - 1) % stevilo_vseh_zvezd + 1

# 4. Diferencialna fotometrija (ID 2 je referenčna zvezda)
ref_podatki = df[df['PRAVI_ID'] == 2][['IMAGE', 'MAG']].rename(columns={'MAG': 'REF_MAG'})
df = pd.merge(df, ref_podatki, on='IMAGE')

# AG Dra magnituda - ref zvezda + 7.6
df['DIFF_MAG'] = df['MAG'] - df['REF_MAG'] + 7.6

# 5. Izluščimo AG Dra (ID 1), sortiramo in odrežemo zadnjih 50 meritev
ag_dra_df = df[df['PRAVI_ID'] == 1].copy()
ag_dra_df = ag_dra_df.sort_values(by='OTIME')
ag_dra_df = ag_dra_df[:-50]

# 6. Priprava za izvoz in shranjevanje v txt
# Izberemo samo stolpca za čas in diferencialno magnitudo
izvoz_df = ag_dra_df[['OTIME', 'DIFF_MAG']]

# Shranimo v datoteko ločeno s tabulatorjem
ime_izvoza = 'AG_Dra.txt'
izvoz_df.to_csv(ime_izvoza, sep='\t', index=False)

print(f"Datoteka '{ime_izvoza}' je bila uspešno ustvarjena!")


# --------- 7. IZRIS GRAFA ---------

# Čas pretvorimo v datetime za lepši prikaz na X osi
ag_dra_df['URE'] = Time(ag_dra_df['OTIME'].values, format='mjd').datetime

# Izračunamo drseče povprečje (samo za lepši izris, izvozili smo namreč surove)
velikost_okna = 8
ag_dra_df['ROLLING_MAG'] = ag_dra_df['DIFF_MAG'].rolling(window=velikost_okna, center=True).mean()

plt.figure(figsize=(10, 6))

# Surove meritve (sive pike)
plt.plot(ag_dra_df['URE'], ag_dra_df['DIFF_MAG'], 'o', color='gray', alpha=0.6, markersize=3, label='Original')

# Zglajena krivulja (modra črta)
plt.plot(ag_dra_df['URE'], ag_dra_df['ROLLING_MAG'], '-o', color='blue', linewidth=2, markersize=3, label=f'Rolling Avg ({velikost_okna})')

# Nastavitve grafa
plt.gca().invert_yaxis()  # V astronomiji gredo magnitude navzdol (svetlejše zgoraj)
plt.xlabel('Čas (UTC)', fontsize=14)
plt.ylabel('Magnituda', fontsize=14)
plt.title('Svetlobna krivulja AG Dra', fontsize=14, fontweight='bold')
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# Formatiranje ur na X osi
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.show()