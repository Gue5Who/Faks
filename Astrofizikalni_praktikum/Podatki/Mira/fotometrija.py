import os
import glob
from pyraf import iraf

# Naloži potrebne pakete
iraf.noao()
iraf.imred()
iraf.ccdred()
iraf.digiphot()
iraf.apphot()
iraf.ptools() # za pdump

def izvedi_fotometrijo(vhodna_mapa, izhodna_mapa, fwhm=4.0, threshold=10.0, cbox=10.0, annulus=15.0, dannulus=5.0, apertures=8.0, ref_id=None, maxshift = 100):
    """
    Skripta za avtomatizacijo fotometrije z uporabo PyRAF.
    
    Parametri:
    vhodna_mapa   : pot do mape z neobdelanimi fits posnetki (npr. 'Podatki/Mira/')
    izhodna_mapa  : pot do mape, kamor se shranijo obdelani podatki in rezultati
    fwhm          : ocena FWHM zvezde v pikslih (dobiš iz imexam)
    threshold     : prag za iskanje zvezd (daofind) v enotah sigme ozadja
    cbox          : velikost centrirnega polja (centerpars)
    annulus       : notranji radij obroča za oceno ozadja (fitskypars)
    dannulus      : širina obroča za oceno ozadja (fitskypars)
    apertures     : radij aperture za fotometrijo (photpars)
    ref_id        : tekst (ali številka), ki se nahaja v imenu referenčnega posnetka (npr. "001" za "mira_001.fits"). Če je None, vzame prvega.
    """
    
    # Ustvari izhodno mapo, če ne obstaja
    if not os.path.exists(izhodna_mapa):
        os.makedirs(izhodna_mapa)
    
    print(f"Začenjam analizo v območju {vhodna_mapa}")
    print(f"Rezultati bodo shranjeni v {izhodna_mapa}")
    
    # Prestavimo se v mapo s podatki za lažje delo z irafom (ali pa podajamo polne poti)
    stara_pot = os.getcwd()
    os.chdir(vhodna_mapa)
    
    # 1. Združevanje kalibracijskih posnetkov
    print("\n--- Združevanje kalibracijskih posnetkov ---")
    
    masterbias = os.path.join(stara_pot, izhodna_mapa, "masterbias.fits")
    masterdark = os.path.join(stara_pot, izhodna_mapa, "masterdark.fits")
    masterflat = os.path.join(stara_pot, izhodna_mapa, "masterflat.fits")
    
    # Pobriši prejšnje master datoteke, če obstajajo
    for f in [masterbias, masterdark, masterflat]:
        if os.path.exists(f):
            os.remove(f)
            
    # Nastavitev parametrov za ccdred
    iraf.ccdred.instrument = "" # Prepreči iskanje specifičnih headerjev inštrumenta
    
    # Zerocombine
    bias_files = glob.glob("Bias/*.fits") + glob.glob("Bias/*.fit")
    if bias_files:
        print("Izvajam zerocombine...")
        iraf.zerocombine(input="Bias/*.fit*", output=masterbias, combine="median", reject="none", scale="none", ccdtype="")
    else:
        print("Opozorilo: Ni posnetkov bias*")

    # Darkcombine
    dark_files = glob.glob("Dark/*.fits") + glob.glob("Dark/*.fit")
    if dark_files:
        print("Izvajam darkcombine...")
        iraf.darkcombine(input="Dark/*.fit*", output=masterdark, combine="median", reject="none", scale="exposure", ccdtype="", process="no")
    else:
        print("Opozorilo: Ni posnetkov dark*")

    # Flatcombine (pazi če je filter v imenu datoteke)
    flat_files = glob.glob("Flat/*.fits") + glob.glob("Flat/*.fit")
    if flat_files:
        print("Izvajam flatcombine...")
        iraf.flatcombine(input="Flat/*.fit*", output=masterflat, combine="median", reject="none", 
                         process="no", subset="no", delete="no", clobber="no", scale="mode", ccdtype="")
    else:
        print("Opozorilo: Ni posnetkov flat*")

    # 2. Obdelava posnetkov zvezd (ccdproc)
    print("\n--- Obdelava posnetkov (ccdproc) ---")
    
    # Najdi vse posnetke zvezd (v mapi Light)
    posnetki_zvezde = glob.glob("Light/*.fits") + glob.glob("Light/*.fit")
    
    if not posnetki_zvezde:
        print("Napaka: Ni posnetkov zvezd za obdelavo!")
        os.chdir(stara_pot)
        return
        
    lista_surovih = os.path.join(stara_pot, izhodna_mapa, "surovi.list")
    lista_obdelanih = os.path.join(stara_pot, izhodna_mapa, "obdelani.list")
    
    with open(lista_surovih, "w") as f1, open(lista_obdelanih, "w") as f2:
        for posnetek in posnetki_zvezde:
            ime, koncnica = os.path.splitext(os.path.basename(posnetek))
            obdelan = os.path.join(stara_pot, izhodna_mapa, ime + "-o" + koncnica)
            # Pobriši že obstoječe obdelane
            if os.path.exists(obdelan):
                os.remove(obdelan)
                
            f1.write(posnetek + "\n")
            f2.write(obdelan + "\n")
            
    print("Izvajam ccdproc...")
    iraf.ccdproc(images="@" + lista_surovih, output="@" + lista_obdelanih,
                 ccdtype="",
                 noproc="no",
                 fixpix="no", overscan="no", trim="no",
                 zerocor="yes",
                 darkcor="yes",
                 flatcor="yes",
                 zero=masterbias, dark=masterdark, flat=masterflat)

    # 3. Iskanje zvezd na referenčnem posnetku (daofind)
    print("\n--- Iskanje zvezd (daofind) ---")
    
    referencni_posnetek = None
    with open(lista_obdelanih, "r") as f2:
        vsi_obdelani = f2.readlines()
        
    if ref_id is not None:
        for vrstica in vsi_obdelani:
            # Iščemo tekst `ref_id` v imenu posnetka
            if str(ref_id) in vrstica:
                referencni_posnetek = vrstica.strip()
                break
                
    # Če nismo podali ref_id ali pa ga ni našel, vzemi kr prvega
    if referencni_posnetek is None and len(vsi_obdelani) > 0:
        referencni_posnetek = vsi_obdelani[0].strip()
        
    # Naredi coo datoteko v izhodni mapi
    ime_ref = os.path.basename(referencni_posnetek)
    coo_file = os.path.join(stara_pot, izhodna_mapa, ime_ref + ".coo")
    if os.path.exists(coo_file):
        os.remove(coo_file)
        
    # Nastavljanje parametrov daofind (datapars, findpars)
    iraf.datapars.fwhmpsf = fwhm
    iraf.datapars.sigma = 4.0 # Ocena šuma ozadja (fajn spremeniti glede na posnetek)
    iraf.datapars.readnoi = 1.0 # Read noise (če ni znani daj 1)
    iraf.datapars.gain = "EGAIN" # e-/ADU gain
    iraf.datapars.epadu = 0.779999971389771 # e-/ADU gain
    # exposure, filter, obstime se preberejo iz headerja
    # 
    # PREVERI HEADERJE
    iraf.datapars.exposure = "EXPOSURE" 
    iraf.datapars.airmass = "AIRMASS"
    iraf.datapars.filter = "FILTER"
    iraf.datapars.obstime = "MJD-OBS"
     
    iraf.findpars.threshold = threshold
    
    print(f"Iščem zvezde na referenčnem posnetku: {referencni_posnetek}")
    iraf.daofind(image=referencni_posnetek, output=coo_file,
                 interactive="no", verify="no")

    # 4. Fotometrija (phot)
    print("\n--- Fotometrija (phot) ---")
    
    # Nastavitve za phot
    iraf.centerpars.calgorithm = "centroid"
    iraf.centerpars.cbox = cbox
    iraf.centerpars.maxshift = maxshift
    
    iraf.fitskypars.salgorithm = "mode"
    iraf.fitskypars.annulus = annulus
    iraf.fitskypars.dannulus = dannulus
    
    iraf.photpars.apertures = apertures
    #iraf.photpars.zmag = 25.0 # Zero point (lahko spremeniš)
    
    datoteka_mag = os.path.join(stara_pot, izhodna_mapa, "fotometrija_vse.mag")
    if os.path.exists(datoteka_mag):
        os.remove(datoteka_mag)
        
    print("Izvajam phot na vseh obdelanih posnetkih...")
    # iraf.phot potrebuje za coords nek file z xy koordinatami
    # tukaj bomo uporabili coords=coo_file is referencne slike za vse obdelane
    iraf.phot(image="@" + lista_obdelanih, coords=coo_file, output=datoteka_mag,
              interactive="no", verify="no")

    # 5. Ekstrakcija podatkov (pdump)
    print("\n--- Ekstrakcija podatkov (pdump) ---")
    
    rezultat_txt = os.path.join(stara_pot, izhodna_mapa, "rezultati_fotometrija.txt")
    if os.path.exists(rezultat_txt):
        os.remove(rezultat_txt)
        
    # Količine, ki jih hočemo (npr. ime slike, x, y, mag, napaka, zračna masa, filter, JD)
    # Zamenjaj glede na potrebe za tvoj HR diagram / svetlobno krivuljo
    polja_za_izpis = "IMAGE,XCENTER,YCENTER,MAG,MERR,IFILTER,OTIME,ID"
    
    print(f"Zapisujem {polja_za_izpis} v {rezultat_txt}")
    
    iraf.pdump(infiles=datoteka_mag, fields=polja_za_izpis, expr= "PERROR='NoError' & CERROR='NoError' & SERROR='NoError'", Stdout=rezultat_txt)
    
    # Vrnitev nazaj v začetno mapo
    os.chdir(stara_pot)
    print("\nFotometrija končana! Vsi obdelani posnetki in datoteke so v mapi:")
    print(os.path.abspath(izhodna_mapa))

# Primer klica, prilagodi poti in parametre
if __name__ == "__main__":
    # Poti
    mapa_podatkov = "." # Uporabi trenutno mapo, ker so podatki tukaj v Bias, Dark, Flat, Light
    mapa_rezultatov = "Obdelani_podatki"
    
    # parametri iz tvojega predogleda (imexam)
    fwhm = 15          # FWHM zvezde
    threshold = 50.0    # prag v daofind
    cbox = 45.0         # širina okvirčka za centriranje = cbox
    maxshift = 100      # za koliko se premakne
    annulus = 16.0      # začetek obroča ozadja (nekaj več kot fwhm, npr r+2)
    dannulus = 5.0      # širina obroča ozadja
    apertures = 15.0     # radij notranjega kroga (apertura zvezde)
    ref_id = "0516"       # Ime ali številka referenčnega posnetka (npr. "001"). Če želiš kr prvega, nastavi na None.
    
    print("Začetek skripte...")
    
    izvedi_fotometrijo(
        vhodna_mapa=mapa_podatkov, 
        izhodna_mapa=mapa_rezultatov,
        fwhm=fwhm,
        threshold=threshold,
        cbox=cbox,
        maxshift = maxshift,
        annulus=annulus,
        dannulus=dannulus,
        apertures=apertures,
        ref_id=ref_id
    )
