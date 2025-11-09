import numpy as np
import matplotlib.pyplot as plt
from galpy.orbit import Orbit
import astropy.units as u
from galpy.potential import MWPotential2014, vcirc
from galpy.potential.McMillan17 import McMillan17

def sample_exponential_disk(N, lR, lz, Rmax=None):
    """Sample stars from a double-exponential disk."""
    R = np.random.gamma(shape=2.0, scale=lR.value, size=N) * lR.unit
    z = np.random.choice([-1, 1], size=N) * np.random.exponential(lz.value, size=N) * lz.unit
    phi = np.random.uniform(0, 2*np.pi, size=N) * u.rad

    if Rmax is not None:
        mask = R < Rmax
        R, z, phi = R[mask], z[mask], phi[mask]

    x = R * np.cos(phi)
    y = R * np.sin(phi)
    return x, y, z


def sample_spherical_component(N, lR, Rmax=None):
    """Sample stars from a spherical exponential component (ρ ∝ e^{-R/lR})."""
    R = np.random.gamma(shape=3.0, scale=lR.value, size=N) * lR.unit
    if Rmax is not None:
        R = R[R < Rmax]

    phi = np.random.uniform(0, 2*np.pi, len(R))
    cos_theta = np.random.uniform(-1, 1, len(R))
    sin_theta = np.sqrt(1 - cos_theta**2)

    x = R * sin_theta * np.cos(phi)
    y = R * sin_theta * np.sin(phi)
    z = R * cos_theta
    return x, y, z


params = {"halo":  {"lR": 15.0*u.kpc,"N": 100000, "color": "yellow"}}


Rmax_halo  = 50 * u.kpc


#  Generate all components
x_halo,  y_halo,  z_halo  = sample_spherical_component(params["halo"]["N"],  params["halo"]["lR"],  Rmax=Rmax_halo)

# --- Common setup ---
ro = 8.0 * u.kpc
vo = 220.0 * u.km/u.s
t_end = 4 * u.Gyr
n_steps = 5000
ts = np.linspace(0, t_end.value, n_steps) * u.Gyr


pot = MWPotential2014

print(f"✅  Potential ready for orbit integration.")

# --- Parameters for halo ---
sigma_R, sigma_T, sigma_z = 120, 120, 120  # km/s
mean_vT = 0                                # no rotation

x0, y0, z0 = x_halo.value, y_halo.value, z_halo.value
R0 = np.sqrt(x0**2 + y0**2)
phi0 = np.arctan2(y0, x0)
N = len(x0)

vR0 = np.random.normal(0, sigma_R, N) * u.km/u.s
vT0 = np.random.normal(mean_vT, sigma_T, N) * u.km/u.s
vz0 = np.random.normal(0, sigma_z, N) * u.km/u.s

orbits_halo = [
    Orbit([R0[i]*u.kpc, vR0[i], vT0[i], z0[i]*u.kpc, vz0[i], phi0[i]*u.rad],
          ro=ro, vo=vo)
    for i in range(N)
]

for o in orbits_halo:
    o.integrate(ts, pot, method='symplec4_c')

print(f"✅ Integrated {N} halo orbits for {t_end}.")

    # Helper functions to handle both unitful and unitless outputs
def safe_val(val, scale=1.0):
    """Return numeric value from a Galpy output (Quantity or float)."""
    if hasattr(val, "to"):
        return val.value
    else:
        return np.array(val) * scale

# Use ro (kpc) and vo (km/s) for scaling
ro_val = ro.value if hasattr(ro, "value") else ro
vo_val = vo.value if hasattr(vo, "value") else vo


x_i = np.array([safe_val(o.x(ts[0]), ro_val) for o in orbits_halo])
y_i = np.array([safe_val(o.y(ts[0]), ro_val) for o in orbits_halo])
z_i = np.array([safe_val(o.z(ts[0]), ro_val) for o in orbits_halo])

vx_i = np.array([safe_val(o.vx(ts[0]), vo_val) for o in orbits_halo])
vy_i = np.array([safe_val(o.vy(ts[0]), vo_val) for o in orbits_halo])
vz_i = np.array([safe_val(o.vz(ts[0]), vo_val) for o in orbits_halo])

x_f = np.array([safe_val(o.x(ts[-1]), ro_val) for o in orbits_halo])
y_f = np.array([safe_val(o.y(ts[-1]), ro_val) for o in orbits_halo])
z_f = np.array([safe_val(o.z(ts[-1]), ro_val) for o in orbits_halo])

vx_f = np.array([safe_val(o.vx(ts[-1]), vo_val) for o in orbits_halo])
vy_f = np.array([safe_val(o.vy(ts[-1]), vo_val) for o in orbits_halo])
vz_f = np.array([safe_val(o.vz(ts[-1]), vo_val) for o in orbits_halo])

data = np.column_stack([x_i, y_i, z_i, vx_i, vy_i, vz_i, x_f, y_f, z_f, vx_f, vy_f, vz_f])
header = "x_i[kpc] y_i[kpc] z_i[kpc] vx_i[km/s] vy_i[km/s] vz_i[km/s] x_f[kpc] y_f[kpc] z_f[kpc] vx_f[km/s] vy_f[km/s] vz_f[km/s]"
np.savetxt(f"orbits_halo_{t_end.value}Gyrs.txt", data, header=header, fmt="%.6f")

print(f"✅ Saved {N} halo orbits to orbits_halo.txt")