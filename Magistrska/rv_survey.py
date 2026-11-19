import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from astroquery.xmatch import XMatch

# Seznam zvezd iz vaše mape (skupaj 42 zvezd)
CILJNE_ZVEZDE = {
    'A2-001', 'A2-006', 'A2-010', 'A2-011', 'A2-023', 'A2-025', 'A2-028', 'A2-030',
    'A2-045', 'A2-052', 'A2-053', 'A2-075', 'A2-077', 'A2-081', 'A2-085', 'A2-091',
    'A2-103', 'A2-106', 'A2-160', 'A2-164', 'A2-171', 'A2-178', 'A2-184', 'A2-206',
    'A2-207', 'A2-216', 'A2-222', 'A2-224', 'A2-231', 'A2-234', 'A2-235', 'A2-236',
    'A2-249', 'A2-250', 'A2-251', 'A2-256', 'A2-292', 'A2-323', 'A2-342', 'A2-354',
    'A2-356', 'A2-362'
}

def nalozi_in_filtriraj_koordinate(filename, filter_set=None):
    ids, ras, decs = [], [], []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('---'):
                continue
            
            parts = line.split()
            if len(parts) >= 3:
                obj_id = parts[0]
                
                # Če imamo nastavljen filter, obdrži le tiste zvezde s seznama
                if filter_set is not None:
                    if obj_id.upper() not in filter_set:
                        continue
                
                ids.append(obj_id)
                ras.append(parts[1])
                decs.append(parts[2])

    coords = SkyCoord(ras, decs, unit=(u.hourangle, u.deg), frame='icrs')
    
    tabela = Table()
    tabela['ID'] = ids
    tabela['RA_orig'] = ras
    tabela['Dec_orig'] = decs
    tabela['ra_deg'] = coords.ra.deg
    tabela['dec_deg'] = coords.dec.deg
    return tabela

def najdi_stolpec(df, mozna_imena):
    for ime in mozna_imena:
        if ime in df.columns:
            return ime
    return None

def main():
    vhodna_datoteka = 'A2_list_all.txt'
    print(f"1. Berem {vhodna_datoteka} in filtriram le izbrane objekte...")
    tabela_zvezd = nalozi_in_filtriraj_koordinate(vhodna_datoteka, filter_set=CILJNE_ZVEZDE)
    print(f"   Za poizvedbo bo uporabljenih {len(tabela_zvezd)} zvezd.")

    # ---------------------------------------------------------
    # 2. Crossmatch z GAIA DR3
    # ---------------------------------------------------------
    print("\n2. Crossmatch z Gaia DR3 (I/355/gaiadr3)...")
    rezultat_gaia = XMatch.query(
        cat1=tabela_zvezd,
        cat2='vizier:I/355/gaiadr3',
        max_distance=2 * u.arcsec,
        colRA1='ra_deg',
        colDec1='dec_deg'
    )
    df_gaia = rezultat_gaia.to_pandas()
    print(f"   Ujemanje najdeno za {len(df_gaia)} zvezd.")

    col_gaia_rv = najdi_stolpec(df_gaia, ['RV', 'radial_velocity'])
    col_gaia_err = najdi_stolpec(df_gaia, ['e_RV', 'radial_velocity_error'])

    df_osnova = tabela_zvezd.to_pandas()

    if not df_gaia.empty:
        stolpci = ['ID', 'Source', 'angDist']
        if col_gaia_rv: stolpci.append(col_gaia_rv)
        if col_gaia_err: stolpci.append(col_gaia_err)
        
        df_gaia_sub = df_gaia[stolpci].rename(columns={
            'Source': 'Gaia_DR3_ID',
            'angDist': 'sep_Gaia [arcsec]',
            col_gaia_rv: 'RV_Gaia [km/s]' if col_gaia_rv else 'RV_Gaia',
            col_gaia_err: 'e_RV_Gaia [km/s]' if col_gaia_err else 'e_RV_Gaia'
        })
        df_koncna = pd.merge(df_osnova, df_gaia_sub, on='ID', how='left')
    else:
        df_koncna = df_osnova

    # ---------------------------------------------------------
    # 3. Crossmatch z GALAH DR3
    # ---------------------------------------------------------
    print("\n3. Crossmatch z GALAH DR3 (J/MNRAS/506/150/stars)...")
    try:
        rezultat_galah = XMatch.query(
            cat1=tabela_zvezd,
            cat2='vizier:J/MNRAS/506/150/stars',
            max_distance=2 * u.arcsec,
            colRA1='ra_deg',
            colDec1='dec_deg'
        )
        df_galah = rezultat_galah.to_pandas()
        print(f"   Ujemanje v GALAH najdeno za {len(df_galah)} zvezd.")
        
        col_galah_rv = najdi_stolpec(df_galah, ['RVgalah', 'RV', 'rv_galah'])
        col_galah_err = najdi_stolpec(df_galah, ['e_RVgalah', 'e_RV', 'e_rv_galah'])
        if col_galah_rv is None:
            print("   POZOR: RV stolpec ni najden! Stolpci:", list(df_galah.columns))

        if not df_galah.empty:
            stolpci_g = ['ID', 'angDist']
            if col_galah_rv: stolpci_g.append(col_galah_rv)
            if col_galah_err: stolpci_g.append(col_galah_err)
            
            df_galah_sub = df_galah[stolpci_g].rename(columns={
                'angDist': 'sep_GALAH [arcsec]',
                col_galah_rv: 'RV_GALAH [km/s]' if col_galah_rv else 'RV_GALAH',
                col_galah_err: 'e_RV_GALAH [km/s]' if col_galah_err else 'e_RV_GALAH'
            })
            df_koncna = pd.merge(df_koncna, df_galah_sub, on='ID', how='left')
        else:
            print("   (Nobena od teh 42 zvezd nima spektra v GALAH DR3)")

    except Exception as e:
        print(f"   Opozorilo pri GALAH poizvedbi: {e}")

    # ---------------------------------------------------------
    # 4. Shrani v CSV
    # ---------------------------------------------------------
    izhod = 'zvezde_RV_iz_literature.csv'
    df_koncna.to_csv(izhod, index=False)
    print(f"\nKončano! Rezultati so shranjeni v: {izhod}")

    # Prikaz na zaslonu
    prikazi = [c for c in ['ID', 'RV_Gaia [km/s]', 'e_RV_Gaia [km/s]', 'RV_GALAH [km/s]'] if c in df_koncna.columns]
    print("\nIzpis rezultatov za izbrane zvezde:")
    print(df_koncna[prikazi].to_string(index=False))

if __name__ == '__main__':
    main()