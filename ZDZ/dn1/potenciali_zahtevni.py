from galpy.orbit import Orbit
from galpy.potential import MWPotential2014, DehnenBarPotential, LogarithmicHaloPotential, MiyamotoNagaiPotential, IsochronePotential
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from galpy.potential import IsochronePotential
import astropy
from scipy.signal import find_peaks

def Banana_Orbit():
    pot = MWPotential2014 + [DehnenBarPotential(barfreq=1.85, barphi=25*u.deg)]

    o = Orbit(vxvv=[4.0*u.kpc, 0*u.km/u.s, 160*u.km/u.s, 0*u.kpc, 0*u.km/u.s, 0*u.deg])
    ts = np.linspace(0, 4, 20000) * u.Gyr
    o.integrate(ts, pot)

    plt.figure(figsize=(7,7))
    o.plot(d1='x', d2='y')
    plt.title("Banana Orbit (with bar potential)")
    plt.axis('equal')
    plt.show()

def Box_Orbit():
    pot = LogarithmicHaloPotential(q=0.8, normalize=1.0)

    o = Orbit(vxvv=[0.3*u.kpc, 50*u.km/u.s, 0*u.km/u.s, 0*u.kpc, 0*u.km/u.s, 0*u.deg])
    ts = np.linspace(0, 1, 5000) * u.Gyr
    o.integrate(ts, pot)

    #plt.figure(figsize=(7,7))
    o.plot(d1='x', d2='y')
    plt.title("Box Orbit (triaxial potential)")
    #plt.axis('equal')
    plt.show()

def Loop_Orbit():
    pot = MiyamotoNagaiPotential(a=6.5, b=0.26, normalize=1.0)

    o = Orbit(vxvv=[7*u.kpc, 0*u.km/u.s, 230*u.km/u.s, 0*u.kpc, 0*u.km/u.s, 0*u.deg])
    ts = np.linspace(0, 1, 5000) * u.Gyr
    o.integrate(ts, pot)

    #plt.figure(figsize=(7,7))
    o.plot(d1='x', d2='y')
    plt.title("Loop Orbit (Disk potential)")
    plt.axis('equal')
    plt.show()

def Fish_Orbit():
    pot = IsochronePotential(b=0.8, normalize=1.0)

    o = Orbit(vxvv=[1.0*u.kpc, 0*u.km/u.s, 0.85*220*u.km/u.s, 0*u.kpc, 0*u.km/u.s, 0*u.deg])
    ts = np.linspace(0, 2, 40000) * u.Gyr
    o.integrate(ts, pot)

    #plt.figure(figsize=(7,7))
    o.plot(d1='x', d2='y')
    plt.title("Fish Orbit (1:2 resonance)")
    plt.axis('equal')
    plt.show()

# === Pretty Orbits Finder (Galpy) ===
# Finds & plots: 3:2 rosette, 2:1 fish (axisymmetric), and a banana orbit in a rotating bar.
# Copy–paste into Jupyter. Requires: galpy, astropy, matplotlib, numpy

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from galpy.orbit import Orbit
from galpy.potential import IsochronePotential, MWPotential2014, DehnenBarPotential
from astropy.visualization import quantity_support
quantity_support()

# ---------- Helpers ----------
def integrate_orbit(o, pot, tmax_Gyr=10.0, nsteps=20000):
    ts = np.linspace(0, tmax_Gyr, nsteps) * u.Gyr
    o.integrate(ts, pot)
    return ts


def radial_zero_crossings(R):
    """Return indices of local minima of the radius array R."""
    # Convert Quantity -> plain array if needed
    if hasattr(R, "value"):
        R = R.value
    R = np.asarray(R, dtype=float)
    # Find peaks of -R (minima of R)
    peaks, _ = find_peaks(-R)
    return peaks

def estimate_ratio(o, pot, ts):
    """Estimate number of radial oscillations and azimuthal rotations."""
    R = o.R(ts)
    phi = o.phi(ts)
    peaks = radial_zero_crossings(R)
    n_rad = len(peaks)
    # Convert phi to a plain array in radians
    if hasattr(phi, "to"):
        phi = phi.to(u.rad).value
    dphi = phi[-1] - phi[0]
    n_azi = dphi / (2 * np.pi)
    return n_rad, n_azi, dphi


def search_resonance(pot, R0=1.0*u.kpc, vT_grid=None, vr0=0*u.km/u.s, target=(3,2),
                     tmax_Gyr=10.0, nsteps=20000):
    if vT_grid is None:
        # scan 0.6–1.5 of 220 km/s
        vT_grid = np.linspace(0.6, 1.5, 50)*220*u.km/u.s
    best = None
    for vT in vT_grid:
        o = Orbit(vxvv=[R0, vr0, vT, 0*u.kpc, 0*u.km/u.s, 0*u.deg])
        ts = integrate_orbit(o, pot, tmax_Gyr=tmax_Gyr, nsteps=nsteps)
        n_rad, n_azi, _ = estimate_ratio(o, pot, ts)
        if n_rad < 4:  # too few cycles; skip
            continue
        ratio = n_rad / max(n_azi, 1e-6)
        target_ratio = target[0]/target[1]
        err = abs(ratio - target_ratio)/target_ratio
        if (best is None) or (err < best['err']):
            best = dict(o=o, ts=ts, vT=vT, ratio=ratio, err=err, n_rad=n_rad, n_azi=n_azi)
    return best

def plot_xy(o, ts, title):
    """Plot orbit in the x–y plane, unit-safe."""
    x = o.x(ts)
    y = o.y(ts)
    # Convert to kpc if quantities, else just arrays
    if hasattr(x, "to"):
        x = x.to(u.kpc).value
    if hasattr(y, "to"):
        y = y.to(u.kpc).value
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    plt.figure(figsize=(6.6, 6.6))
    plt.plot(x, y, lw=1.2)
    plt.gca().set_aspect("equal", "box")
    plt.xlabel("x [kpc]")
    plt.ylabel("y [kpc]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

# ---------- 1) Three-lobed rosette (3:2) in a soft axisymmetric potential ----------
iso = IsochronePotential(b=0.8, normalize=1.0)  # “soft” potential -> clean rosettes
best_32 = search_resonance(
    pot=iso,
    R0=1.0*u.kpc,
    vT_grid=np.linspace(0.8, 1.4, 60)*220*u.km/u.s,
    target=(3,2),
    tmax_Gyr=12.0,
    nsteps=30000
)
print("[3:2] vT ~", best_32['vT'], "  ratio n_rad/n_azi ~", best_32['ratio'])
plot_xy(best_32['o'], best_32['ts'], "Near 3:2 Rosette (Isochrone)")

# ---------- 2) Fish (2:1) in the same axisymmetric potential ----------
best_21 = search_resonance(
    pot=iso,
    R0=1.0*u.kpc,
    vT_grid=np.linspace(0.6, 1.2, 60)*220*u.km/u.s,
    target=(2,1),
    tmax_Gyr=16.0,
    nsteps=35000
)
print("[2:1] vT ~", best_21['vT'], "  ratio n_rad/n_azi ~", best_21['ratio'])
plot_xy(best_21['o'], best_21['ts'], "Near 2:1 “Fish” (Isochrone)")

# ---------- 3) Banana orbit in a rotating bar (plot in the bar’s rotating frame) ----------
# Important: these appear clearly only in the *rotating frame* of the bar.
# We'll build MWPotential2014 + DehnenBar and transform x,y -> bar frame x',y' by rotating by -Omega_bar * t.

# Natural units: ro, vo set by Orbit; we'll use the defaults vo=220 km/s, ro=8 kpc unless you change them on the Orbit.
# Pattern speed Omega_bar: choose ~40 km/s/kpc (typical MW-ish). Convert to natural units (vo/ro).
vo = 220*u.km/u.s
ro = 8.0*u.kpc
Omega_bar_phys = 40.0 * (u.km/u.s) / u.kpc   # ~40 km/s/kpc
Omega_bar_natural = (Omega_bar_phys / (vo/ro)).decompose().value  # dimensionless for galpy
bar = DehnenBarPotential(omegab=Omega_bar_natural, barphi=25*u.deg, rb=0.8, Af=0.01)

pot_barred = MWPotential2014 + [bar]

# Pick an orbit inside the bar region and integrate long enough
o_ban = Orbit(vxvv=[3.5*u.kpc, 0*u.km/u.s, 140*u.km/u.s, 0*u.kpc, 0*u.km/u.s, 0*u.deg], ro=ro, vo=vo)
ts_b = integrate_orbit(o_ban, pot_barred, tmax_Gyr=4.0, nsteps=30000)

# Transform to the *rotating frame* of the bar: (x',y') = R(-Omega_bar * t) * (x,y)
# Need Omega_bar in rad/Gyr to rotate the coordinates taken at times ts_b
# Convert 1 km/s/kpc -> rad/Gyr: 1 km/s/kpc ≈ 1.022712 rad/Gyr
conv = 1.022712
Omega_bar_rad_per_Gyr = (Omega_bar_phys.to_value((u.km/u.s)/u.kpc)) * conv  # rad/Gyr

# --- Banana orbit: transform to the rotating bar frame safely ---

# Extract positions (handle Quantity or ndarray)
x = o_ban.x(ts_b)
y = o_ban.y(ts_b)
if hasattr(x, "to"):
    x = x.to(u.kpc).value
if hasattr(y, "to"):
    y = y.to(u.kpc).value
x = np.asarray(x, dtype=float)
y = np.asarray(y, dtype=float)

# Convert time to Gyr for rotation
if hasattr(ts_b, "to"):
    t_Gyr = ts_b.to(u.Gyr).value
else:
    t_Gyr = np.asarray(ts_b, dtype=float)

# Rotation by -Omega_bar * t to bar frame
conv = 1.022712  # 1 km/s/kpc = 1.022712 rad/Gyr
Omega_bar_rad_per_Gyr = Omega_bar_phys.to_value((u.km/u.s)/u.kpc) * conv
theta = -Omega_bar_rad_per_Gyr * t_Gyr

c, s = np.cos(theta), np.sin(theta)
xp = c * x - s * y
yp = s * x + c * y

# Plot orbit in bar frame
plt.figure(figsize=(6.6, 6.6))
plt.plot(xp, yp, lw=1.2)
plt.gca().set_aspect("equal", "box")
plt.xlabel("x' [kpc] (bar frame)")
plt.ylabel("y' [kpc] (bar frame)")
plt.title("Banana Orbit (Rotating Bar Frame)")
plt.grid(True, alpha=0.3)
plt.show()
