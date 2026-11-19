import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit

# Konstante
C_KMS = 299792.458 # Hitrost svetlobe v km/s
W_D2 = 5889.950    # Rest valovna dolžina Na I D2
W_D1 = 5895.924    # Rest valovna dolžina Na I D1

def get_instrument_resolution(header):
    """
    Samodejno prepozna instrument in vrne resolucijo.
    Asiago (CDELT1 ~ 0.09) -> R ~ 24000
    Varese (CDELT1 ~ 0.21) -> R ~ 11000
    """
    cdelt = header.get('CDELT1', header.get('CD1_1', 0))
    if cdelt > 0 and cdelt < 0.15:
        return 24000 # Asiago
    elif cdelt >= 0.15:
        return 11000 # Varese
    return 20000

def ism_absorption(wave, v_rad, b_kms, tau_d2, tau_d1, R_instrument):
    fwhm_inst = C_KMS / R_instrument
    sig_inst = fwhm_inst / 2.355
    sig_total_kms = np.sqrt(b_kms**2 + sig_inst**2)
    sig_wave_d2 = W_D2 * (sig_total_kms / C_KMS)
    sig_wave_d1 = W_D1 * (sig_total_kms / C_KMS)
    c_d2 = W_D2 * (1.0 + v_rad / C_KMS)
    c_d1 = W_D1 * (1.0 + v_rad / C_KMS)
    tau = tau_d2 * np.exp(-0.5 * ((wave - c_d2) / sig_wave_d2)**2) + \
          tau_d1 * np.exp(-0.5 * ((wave - c_d1) / sig_wave_d1)**2)
    return np.exp(-tau)

def double_gaussian(w, v1, b1, td2_1, td1_1, v2, b2, td2_2, td1_2, R_instrument):
    return ism_absorption(w, v1, b1, td2_1, td1_1, R_instrument) * ism_absorption(w, v2, b2, td2_2, td1_2, R_instrument)

def process_spectrum(fits_path, base_out_dir="Rezultati"):
    print(f"\n--- Obdelujem: {fits_path} ---")
    
    # 1. Priprava map
    rel_path = os.path.relpath(fits_path, start="Paper_II")
    out_dir = os.path.join(base_out_dir, os.path.dirname(rel_path))
    os.makedirs(out_dir, exist_ok=True)
    
    file_basename = os.path.basename(fits_path).replace(".fits", "")
    out_plot = os.path.join(out_dir, f"{file_basename}_fit.png")
    out_data = os.path.join(out_dir, f"{file_basename}_data.txt")
    
    # 2. Branje FITS
    with fits.open(fits_path) as hdul:
        header = hdul[0].header
        flux = hdul[0].data
        crval = header.get('CRVAL1', 0)
        cdelt = header.get('CDELT1', header.get('CD1_1', 1)) 
        crpix = header.get('CRPIX1', 1)
        pixels = np.arange(len(flux))
        wavelengths = crval + (pixels + 1 - crpix) * cdelt
        
    R_inst = get_instrument_resolution(header)
    
    mask = (wavelengths > 5830) & (wavelengths < 5960)
    x = wavelengths[mask]
    y = flux[mask]
    
    mask_local = (x > 5880) & (x < 5905)
    med_val = np.median(y[mask_local])
    y_norm = y / med_val

    # Fittanje Dvojnega Gaussa
    def model_func(w, v1, b1, td2_1, td1_1, v2, b2, td2_2, td1_2):
        return double_gaussian(w, v1, b1, td2_1, td1_1, v2, b2, td2_2, td1_2, R_inst)

    p0 = [-25.0, 3.0, 1.0, 0.5, 0.0, 8.0, 1.0, 0.5]
    bounds = ([-100, 0.1, 0, 0, -100, 0.1, 0, 0], [100, 30, 10, 10, 100, 30, 10, 10])
    
    try:
        popt, pcov = curve_fit(model_func, x, y_norm, p0=p0, bounds=bounds)
        final_flux = model_func(x, *popt)
        
        # Izračun napak fitta iz kovariančne matrike
        # Če je pcov inf, nastavimo na 0
        if np.isinf(pcov).any() or np.isnan(pcov).any():
            fit_errs = np.zeros(8)
        else:
            fit_errs = np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"  OPOZORILO: Fittanje je odpovedalo ({e}).")
        popt = p0
        fit_errs = np.zeros(8)
        final_flux = np.ones_like(x)
        
    v1_rad, b1_kms, tau_d2_1, tau_d1_1, v2_rad, b2_kms, tau_d2_2, tau_d1_2 = popt
    e_v1, e_b1, e_td2_1, e_td1_1, e_v2, e_b2, e_td2_2, e_td1_2 = fit_errs

    # ASTROFIZIKALNE NAPAKE (Sistematske napake meritve)
    # 1. Napaka kalibracije hitrosti instrumenta (običajno ~ 1/10 resolucijskega elementa)
    sys_err_v = (C_KMS / R_inst) / 10.0  # Za Asiago ~1.25 km/s, Varese ~2.7 km/s
    
    # 2. Napaka instrumentalnega profila, ki vpliva na širino b (konservativno 1/20 FWHM)
    sys_err_b = (C_KMS / R_inst) / 20.0
    
    # 3. Napaka postavitve kontinuuma, ki vpliva na globino tau (običajno ~2% šuma)
    sys_err_tau = 0.02 

    # CELOTNA NAPAKA (seštevek kvadratov napake fitta in sistematske napake meritve)
    tot_v1 = np.sqrt(e_v1**2 + sys_err_v**2)
    tot_b1 = np.sqrt(e_b1**2 + sys_err_b**2)
    tot_td2_1 = np.sqrt(e_td2_1**2 + sys_err_tau**2)
    tot_td1_1 = np.sqrt(e_td1_1**2 + sys_err_tau**2)
    
    tot_v2 = np.sqrt(e_v2**2 + sys_err_v**2)
    tot_b2 = np.sqrt(e_b2**2 + sys_err_b**2)
    tot_td2_2 = np.sqrt(e_td2_2**2 + sys_err_tau**2)
    tot_td1_2 = np.sqrt(e_td1_2**2 + sys_err_tau**2)

    # 5. Izvoz podatkov (TXT)
    with open(out_data, "w") as f:
        f.write(f"# File: {fits_path}\n")
        f.write(f"# Resolution R: {R_inst}\n")
        f.write(f"# Instrumentalni/Sistematski prispevek k napaki: err_V={sys_err_v:.3f} km/s, err_b={sys_err_b:.3f} km/s, err_tau={sys_err_tau:.3f}\n")
        f.write(f"# ---------------------------------------------------------\n")
        f.write(f"# Fit Params (Value +- FitError | TotalError):\n")
        f.write(f"# Comp 1 (ISM):\n")
        f.write(f"#   V_rad = {v1_rad:7.3f} +- {e_v1:5.3f} | Tot: {tot_v1:5.3f} km/s\n")
        f.write(f"#   b     = {b1_kms:7.3f} +- {e_b1:5.3f} | Tot: {tot_b1:5.3f} km/s\n")
        f.write(f"#   tD2   = {tau_d2_1:7.3f} +- {e_td2_1:5.3f} | Tot: {tot_td2_1:5.3f}\n")
        f.write(f"#   tD1   = {tau_d1_1:7.3f} +- {e_td1_1:5.3f} | Tot: {tot_td1_1:5.3f}\n")
        f.write(f"# Comp 2 (Zvezda):\n")
        f.write(f"#   V_rad = {v2_rad:7.3f} +- {e_v2:5.3f} | Tot: {tot_v2:5.3f} km/s\n")
        f.write(f"#   b     = {b2_kms:7.3f} +- {e_b2:5.3f} | Tot: {tot_b2:5.3f} km/s\n")
        f.write(f"#   tD2   = {tau_d2_2:7.3f} +- {e_td2_2:5.3f} | Tot: {tot_td2_2:5.3f}\n")
        f.write(f"#   tD1   = {tau_d1_2:7.3f} +- {e_td1_2:5.3f} | Tot: {tot_td1_2:5.3f}\n")
        f.write(f"# ---------------------------------------------------------\n")
        f.write("# Wave, Obs_Flux_Raw, Obs_Flux_Norm, Final_Fit_Flux\n")
        for i in range(len(x)):
            f.write(f"{x[i]:.4f} {y[i]:.4f} {y_norm[i]:.4f} {final_flux[i]:.4f}\n")
            
    # 6. Izvoz slike (PNG)
    plt.figure(figsize=(14, 7))
    plt.plot(x, y_norm, color='black', alpha=0.7, label="Opazovanje")
    plt.plot(x, final_flux, color='blue', linewidth=2, label="Dvojni Gaussov Fit")
    
    comp1 = ism_absorption(x, v1_rad, b1_kms, tau_d2_1, tau_d1_1, R_inst)
    comp2 = ism_absorption(x, v2_rad, b2_kms, tau_d2_2, tau_d1_2, R_inst)
    plt.plot(x, comp1, 'g--', linewidth=1.5, label='Komponenta 1 (ISM)', alpha=0.8)
    plt.plot(x, comp2, 'r--', linewidth=1.5, label='Komponenta 2 (Zvezda)', alpha=0.8)
    
    fit_text = (f"Komponenta 1 (ISM plin):\n"
                f"$V_{{rad}}$ = {v1_rad:.1f} $\\pm$ {tot_v1:.1f} km/s\n"
                f"$b$ = {b1_kms:.1f} $\\pm$ {tot_b1:.1f} km/s\n"
                f"$\\tau_{{D2}}$ = {tau_d2_1:.2f} $\\pm$ {tot_td2_1:.2f}\n\n"
                f"Komponenta 2 (Zvezda):\n"
                f"$V_{{rad}}$ = {v2_rad:.1f} $\\pm$ {tot_v2:.1f} km/s\n"
                f"$b$ = {b2_kms:.1f} $\\pm$ {tot_b2:.1f} km/s\n"
                f"$\\tau_{{D2}}$ = {tau_d2_2:.2f} $\\pm$ {tot_td2_2:.2f}")
    
    plt.text(0.02, 0.05, fit_text, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
             
    plt.xlim(5880, 5905) 
    plt.xlabel("Valovna dolžina (Å)")
    plt.ylabel("Normaliziran Fluks")
    plt.title(f"Fit z napakami: {file_basename}")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()

if __name__ == "__main__":
    vse_datoteke = glob.glob("Paper_II/**/*.fits", recursive=True)
    print(f"Našel sem {len(vse_datoteke)} datotek za obdelavo napak.")
    
    for i, fits_path in enumerate(vse_datoteke):
        print(f"[{i+1}/{len(vse_datoteke)}] {os.path.basename(fits_path)}")
        try:
            process_spectrum(fits_path)
        except Exception as e:
            print(f"  NAPAKA: {e}")
            continue
    print("\nGotovo! Napake so izvožene v _data.txt in narisane na PNG slikah.")
