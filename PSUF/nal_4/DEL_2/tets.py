#!/usr/bin/env python3
import os
from netCDF4 import Dataset

DATA_DIR = "data/Z500/Z500_10"

def test_all_files():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".nc")])
    print(f"Najdenih datotek: {len(files)}\n")

    ref_shape = None
    ref_lat = ref_lon = None

    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        print("="*70)
        print(f"Pregledujem: {fname}")

        try:
            ds = Dataset(path, "r")
        except Exception as e:
            print(f"  ❌ Napaka pri odpiranju: {e}")
            continue

        vars_ = list(ds.variables.keys())
        dims_ = {k: ds.variables[k].shape for k in ds.variables}

        print(f"  Spremenljivke: {vars_}")

        # ---- Najdi lat/lon ----
        lat = None
        lon = None
        for cand in ["lat", "latitude", "nav_lat"]:
            if cand in ds.variables:
                lat = ds.variables[cand][:]
                break
        for cand in ["lon", "longitude", "nav_lon"]:
            if cand in ds.variables:
                lon = ds.variables[cand][:]
                break

        if lat is None or lon is None:
            print("  ❌ Ne najdem lat/lon!")
        else:
            print(f"  lat shape = {lat.shape}, lon shape = {lon.shape}")

        # ---- Najdi polje Z500 ----
        Z_candidates = ["z", "Z", "Z500", "z500", "zg", "gh", "GHT", "hgt"]

        zname = None
        for v in vars_:
            if v in Z_candidates:
                zname = v
                break

        if zname is None:
            # poišči generično polje oblike (time,*,lat,lon)
            for v in vars_:
                shp = ds.variables[v].shape
                if len(shp) >= 3 and shp[-1] == len(lon) and shp[-2] == len(lat):
                    if v not in ["lat", "lon", "time"]:
                        zname = v
                        print(f"  ⚠ Uporabljam '{v}' kot Z500 kandidat.")
                        break

        if zname is None:
            print("  ❌ Ne najdem Z500 kandidata!")
            ds.close()
            continue

        z = ds.variables[zname]
        print(f"  Z500 kandidat: '{zname}' shape = {z.shape}")

        # ---- Pretvori v 3D (time,lat,lon), če možno ----
        shape = z.shape

        if len(shape) == 3:
            z_tll = shape

        elif len(shape) == 4:
            # scenario: (time,1,lat,lon)
            if shape[1] == 1:
                z_tll = (shape[0], shape[2], shape[3])
            # scenario: (1,time,lat,lon)
            elif shape[0] == 1:
                z_tll = (shape[1], shape[2], shape[3])
            # scenario: more plev levels → izberi pidx 0
            else:
                z_tll = (shape[0], shape[2], shape[3])
                print("  ⚠ Več plev nivojev? Uporabil bi 0. nivo.")

        else:
            print("  ❌ Nepričakovan shape Z500:", shape)
            ds.close()
            continue

        print(f"  Normalizirana oblika Z500 = {z_tll}")

        # ---- Primerja oblike med datotekami ----
        if ref_shape is None:
            ref_shape = z_tll
            ref_lat = len(lat)
            ref_lon = len(lon)
        else:
            if z_tll != ref_shape:
                print(f"  ❌ OPOZORILO: Z500 shape se ne ujema s prejšnjimi: {ref_shape} vs {z_tll}")
            if len(lat) != ref_lat:
                print(f"  ❌ lat dimenzija ne ustreza: {ref_lat} vs {len(lat)}")
            if len(lon) != ref_lon:
                print(f"  ❌ lon dimenzija ne ustreza: {ref_lon} vs {len(lon)}")

        ds.close()

    print("\n✓ Testiranje končano.")


if __name__ == "__main__":
    test_all_files()
