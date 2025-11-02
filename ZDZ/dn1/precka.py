# ===========================================
# Dodatna naloga: vpliv prečke na orbite
# ===========================================
from galpy.potential import MWPotential2014, DehnenBarPotential
from galpy.orbit import Orbit
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

# -------------------------------------------
# Čas integracije
# -------------------------------------------
ts = np.linspace(0, 5, 10000) * u.Gyr

def navadna():
    # -------------------------------------------
    # Definicija prečke (Dehnen bar potential)
    # -------------------------------------------
    # --- Physical constants and parameters ---
    G_val = 4.3009e-6  # G in (kpc/M_sun) * (km/s)^2
    M_bar_phys = 1.0e10  # Bar mass [M_sun]
    ro_val = 8.0  # Reference radius [kpc]
    vo_val = 220.0  # Circular velocity [km/s]

    # --- Convert to Galpy natural units ---
    amp_nat = (G_val * M_bar_phys) / (ro_val * vo_val**2)
    omegab_phys = 40.0  # km/s/kpc
    omegab_nat = (omegab_phys * ro_val) / vo_val
    barphi_nat = np.radians(25.0)  # bar angle in radians

    print(f"→ Bar amplitude (natural units): {amp_nat:.4e}")
    print(f"→ Pattern speed (natural units): {omegab_nat:.4f}")
    print(f"→ Bar angle: {barphi_nat:.3f} rad")

    # --- Define Dehnen bar potential ---
    pot_bar = DehnenBarPotential(
        amp=amp_nat,
        omegab=omegab_nat,
        barphi=barphi_nat,
        rb=3.5,           # characteristic bar radius [kpc]
        tform=0.0,        # bar active from start
        tsteady=99.0      # constant over time
    )

    # --- Patch _af if needed (for Galpy bug) ---
    if pot_bar._af is None:
        print("⚠️ Applying _af patch (Galpy internal bug)...")
        pot_bar._af = 1.0


    # -------------------------------------------
    # Kombinirani potencial: MWPotential2014 + prečka
    # -------------------------------------------
    pot_barred = [p for p in MWPotential2014] + [pot_bar]

    # -------------------------------------------
    # Začetna orbita (npr. Sonce)
    # -------------------------------------------
    R0 = 8.0 * u.kpc
    V0 = 220.0 * u.km/u.s
    o_nobar = Orbit([R0, 0.*u.km/u.s, V0, 0.*u.kpc, 0.*u.km/u.s, 0.*u.deg])
    o_bar = o_nobar()

    # -------------------------------------------
    # Integracija
    # -------------------------------------------
    print("Integriram orbito brez prečke...")
    o_nobar.integrate(ts, MWPotential2014, method='leapfrog')

    print("Integriram orbito s prečko...")
    o_bar.integrate(ts, pot_barred, method='leapfrog')

    # -------------------------------------------
    # Izris orbit
    # -------------------------------------------
    plt.figure(figsize=(7, 7))
    plt.plot(o_bar.x(ts), o_bar.y(ts), 'r', lw=1, alpha = 0.5, label='S prečko')
    plt.plot(o_nobar.x(ts), o_nobar.y(ts), 'black', lw=1, label='Brez prečke')
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')
    plt.legend()
    plt.title('Vpliv galaktične prečke na orbito')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def ekscentricna():
    # -----------------------------
    # Potencial z galaktično prečko
    # -----------------------------
    G_val = 4.3009e-6  # (kpc/Msun)*(km/s)^2
    M_bar = 1.0e10     # Msun
    ro_val = 8.0       # kpc
    vo_val = 220.0     # km/s

    amp_nat = (G_val * M_bar) / (ro_val * vo_val**2)
    omegab_nat = (40.0 * ro_val) / vo_val
    barphi_nat = np.radians(25.0)

    pot_bar = DehnenBarPotential(
        amp=amp_nat,
        omegab=omegab_nat,
        barphi=barphi_nat,
        rb=3.5,
        tform=0.0,
        tsteady=99.0
    )

    if pot_bar._af is None:
        pot_bar._af = 1.0

    pot_barred = MWPotential2014 + [pot_bar]

    # -----------------------------
    # Začetni pogoji: bolj ekscentrična orbita
    # -----------------------------
    # Manjša tangencialna hitrost => eliptična orbita
    R0 = 6.0 * u.kpc             # nekoliko bližje središču
    vR0 = 40.0 * u.km/u.s        # radialna komponenta (odmik)
    vT0 = 160.0 * u.km/u.s       # tangencialna hitrost < 220 km/s
    z0 = 0 * u.kpc             # malo iz ravnine
    vz0 = 0* u.km/u.s

    o_bar = Orbit([R0, vR0, vT0, z0, vz0, 0*u.deg])
    o_nobar = o_bar()
    # -----------------------------
    # Integracija orbite
    # -----------------------------
    ts = np.linspace(0, 4, 8000) * u.Gyr
    o_bar.integrate(ts, pot_barred, method='dopr54_c')
    o_nobar.integrate(ts, MWPotential2014, method='dopr54_c')
    

    # -----------------------------
    # Grafični prikaz
    # -----------------------------
    plt.figure(figsize=(7,7))

    plt.plot(o_nobar.x(ts), o_bar.y(ts), 'black', lw=1, alpha = 1, label='Brez prečke')
    plt.plot(o_bar.x(ts), o_bar.y(ts), 'r', lw=1, alpha = 0.5, label='S prečko')
    plt.xlabel("x [kpc]")
    plt.ylabel("y [kpc]")
    plt.title("Ekscentrična orbita")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


ekscentricna()