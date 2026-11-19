import os
import glob
import pandas as pd
from astropy.io import fits
from pathlib import Path

# Tukaj vstavi pot do mape Paper_II na tvojem strežniku
osnovna_pot = "/home/gueswho/Documents/Faks/Magistrska/Paper_II" 

podatki = []

# Uporabimo glob, da poiščemo VSE .fits datoteke v vseh podmapah
vse_fits_datoteke = glob.glob(os.path.join(osnovna_pot, '**', '*.fits'), recursive=True)

print(f"Našel sem {len(vse_fits_datoteke)} FITS datotek. Berem metapodatke...")

for datoteka in vse_fits_datoteke:
    try:
        # 1. IZLUŠČANJE TARČE IZ MAPE
        # Naredimo relativno pot (odstranimo osnovno pot)
        # Npr. iz "Paper_II/A2-001/Asiago/..." dobimo samo "A2-001/Asiago/..."
        relativna_pot = os.path.relpath(datoteka, osnovna_pot)
        
        # Prvi del relativne poti je ime mape, v kateri je tarča
        tarca = relativna_pot.split(os.sep)[0]
        
        # 2. IZLUŠČANJE DATUMA IZ IMENA DATOTEKE
        # Ime datoteke je npr. "20210222_ech_a2-001_hv_c.fits"
        ime_datoteke = os.path.basename(datoteka)
        
        # Vzamemo prvih 8 znakov (YYYYMMDD) in jih lepo formatiramo
        datum_str = ime_datoteke[:8]
        if datum_str.isdigit() and len(datum_str) == 8:
            datum = f"{datum_str[:4]}-{datum_str[4:6]}-{datum_str[6:8]}"
        else:
            datum = "Neznano" # Če se slučajno katera datoteka ne začne z letnico

        # 3. UGOTAVLJANJE OPAZOVALNICE
        opazovalnica = "Neznano"
        if "Asiago" in datoteka or "asiago" in datoteka:
            opazovalnica = "Asiago"
        elif "Varese" in datoteka or "varese" in datoteka:
            opazovalnica = "Varese"

        # 4. BRANJE EKSPOZICIJE IZ FITS GLAVE (tega edino ne moremo iz mape)
        with fits.open(datoteka) as hdul:
            header = hdul[0].header
            ekspozicija = header.get('EXPTIME', 0.0)
                
        # Dodamo v naš seznam
        podatki.append({
            'Tarca': tarca.upper(),  # .upper() pretvori a2-001 v A2-001 za konsistentnost
            'Opazovalnica': opazovalnica,
            'Datum': datum,
            'Ekspozicija_s': ekspozicija,
            'Pot_do_datoteke': datoteka
        })
    except Exception as e:
        print(f"Napaka pri datoteki {datoteka}: {e}")

# Iz seznama ustvarimo pregledno Pandas tabelo
df = pd.DataFrame(podatki)

# Pretvorimo stolpec 'Datum' v pravi format
df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')

# Uredimo tabelo najprej po tarčah (abecedno) in nato kronološko
df = df.sort_values(by=['Tarca', 'Datum'])

# Shranimo
df.to_csv("dnevnik_opazovanj.csv", index=False)

print("\nTabela je posodobljena! Prvih nekaj vrstic:")
print(df.head(10))