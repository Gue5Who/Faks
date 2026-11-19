import os
import glob

C_KMS = 299792.458

def classify_spectra(results_dir="Rezultati"):
    data_files = glob.glob(os.path.join(results_dir, "**", "*_data.txt"), recursive=True)
    
    # Seznami za Asiago (R ~ 24000)
    has_ism_asiago = []
    no_ism_asiago = []
    
    # Seznami za Varese (R ~ 11000)
    has_ism_varese = []
    no_ism_varese = []
    
    for filepath in data_files:
        R_inst = 20000 # default
        v1 = None; b1 = None; tD2_1 = None
        v2 = None
        
        path_lower = filepath.lower()
        if "asiago" in path_lower:
            observatory = "Asiago"
            threshold_inst = C_KMS / 24000
        elif "varese" in path_lower:
            observatory = "Varese"
            threshold_inst = C_KMS / 11000
        else:
            observatory = "Neznano"
            threshold_inst = C_KMS / 20000
            
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("# Comp 1 (ISM):"):
                    # Primer vrstice: # Comp 1 (ISM): V_rad=-33.6, b=0.100, tD2=0.170, tD1=0.000
                    parts = line.split(",")
                    for part in parts:
                        if "V_rad=" in part:
                            v1 = float(part.split("=")[1])
                        elif "b=" in part:
                            b1 = float(part.split("=")[1])
                        elif "tD2=" in part:
                            tD2_1 = float(part.split("=")[1])
                elif line.startswith("# Comp 2 (Zvezda):"):
                    parts = line.split(",")
                    for part in parts:
                        if "V_rad=" in part:
                            v2 = float(part.split("=")[1])
                            
        if v1 is not None and v2 is not None and b1 is not None and tD2_1 is not None:
            # 1. Hitrostni razmak
            diff = abs(v1 - v2)
            
            # 2. Astrofizikalni filtri
            # Pogoj 1: Razmak mora biti večji od instrumentalne ločljivosti
            cond1 = diff > threshold_inst
            
            # Pogoj 2: Širina ISM komponente mora biti fizikalna (optimizator ni zadel meje)
            cond2 = b1 >= 0.5
            
            # Pogoj 3: Globina mora biti netrivialna
            cond3 = tD2_1 >= 0.05
            
            # Pogoj 4: Rayleighov kriterij (razmak > FWHM_ISM, kjer je FWHM = 1.66 * b)
            # Uporabimo manj strog faktor 1.2 (da upoštevamo tudi mešane črte)
            fwhm_ism = 1.66 * b1
            cond4 = diff > fwhm_ism
            
            has_ism = cond1 and cond2 and cond3 and cond4
            
            # Zapišemo razlog, če spektra ni med potrjenimi
            reason = ""
            if not has_ism:
                reasons = []
                if not cond1: reasons.append(f"Razmak {diff:.1f} <= inst. {threshold_inst:.1f}")
                if not cond2: reasons.append(f"Preozko b={b1:.2f}")
                if not cond3: reasons.append(f"Preplitvo tD2={tD2_1:.2f}")
                if not cond4: reasons.append(f"Razmak {diff:.1f} <= FWHM {fwhm_ism:.1f}")
                reason = " | Razlog zavrnitve: " + ", ".join(reasons)
            
            line_out = f"{filepath} (Razlika: {diff:.1f} km/s){reason}"
            
            if observatory == "Asiago":
                if has_ism:
                    has_ism_asiago.append(line_out)
                else:
                    no_ism_asiago.append(line_out)
            elif observatory == "Varese":
                if has_ism:
                    has_ism_varese.append(line_out)
                else:
                    no_ism_varese.append(line_out)
                
    out_file = "Rezultati_ISM_klasifikacija.txt"
    with open(out_file, "w") as f:
        # ASIAGO
        f.write("=================================================================\n")
        f.write(" ASIAGO (Kriteriji: Razmak > 12.5 km/s, b > 0.5, tD2 > 0.05)\n")
        f.write("=================================================================\n")
        f.write(f"\n--- ZAZNAN IN POTRJEN ISM/CSM ({len(has_ism_asiago)} spektrov) ---\n")
        for item in sorted(has_ism_asiago):
            f.write(f"{item}\n")
            
        f.write(f"\n--- ZAVRNJEN ISM/CSM (Zlit ali degeneriran fit) ({len(no_ism_asiago)} spektrov) ---\n")
        for item in sorted(no_ism_asiago):
            f.write(f"{item}\n")
            
        f.write("\n\n")
        
        # VARESE
        f.write("=================================================================\n")
        f.write(" VARESE (Kriteriji: Razmak > 27.3 km/s, b > 0.5, tD2 > 0.05)\n")
        f.write("=================================================================\n")
        f.write(f"\n--- ZAZNAN IN POTRJEN ISM/CSM ({len(has_ism_varese)} spektrov) ---\n")
        for item in sorted(has_ism_varese):
            f.write(f"{item}\n")
            
        f.write(f"\n--- ZAVRNJEN ISM/CSM (Zlit ali degeneriran fit) ({len(no_ism_varese)} spektrov) ---\n")
        for item in sorted(no_ism_varese):
            f.write(f"{item}\n")

    print(f"Pregledanih Asiago: {len(has_ism_asiago) + len(no_ism_asiago)} (Potrjen ISM: {len(has_ism_asiago)}, Zavrnjen: {len(no_ism_asiago)})")
    print(f"Pregledanih Varese: {len(has_ism_varese) + len(no_ism_varese)} (Potrjen ISM: {len(has_ism_varese)}, Zavrnjen: {len(no_ism_varese)})")
    print(f"Nadgrajena klasifikacija shranjena v: {out_file}")

if __name__ == "__main__":
    classify_spectra()
