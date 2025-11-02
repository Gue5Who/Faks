import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric
from galpy.potential import MWPotential2014, DehnenBarPotential
from galpy.potential import vcirc as galpy_vcirc
from galpy.orbit import Orbit
from astropy.coordinates import CartesianDifferential
from scipy.stats import binned_statistic_2d

print("--- Task 3: Generating Galaxy ---")
# This is the base, axisymmetric potential for generating stars
pot_axisymmetric = MWPotential2014
np.random.seed(42)

# --- 1. DEFINE PARAMETERS IN NATURAL UNITS ---
ro = 8.0    # kpc
vo = 220.0  # km/s

# Thin Disk
N_STARS_THIN = 100000
L_R_THIN_NAT = 3.5 / ro
L_Z_THIN_NAT = 0.3 / ro
SIGMA_V_THIN_NAT = 20.0 / vo

# Thick Disk
N_STARS_THICK = 100000
L_R_THICK_NAT = 2.9 / ro
L_Z_THICK_NAT = 1.0 / ro
SIGMA_V_THICK_NAT = 55.0 / vo

# Bulge
N_STARS_BULGE = 50000
L_R_BULGE_NAT = 1.8 / ro
SIGMA_V_BULGE_NAT = 100.0 / vo

# Halo
N_STARS_HALO = 50000
L_R_HALO_NAT = 15.0 / ro
SIGMA_V_HALO_NAT = 120.0 / vo

# --- 2. GENERATE POSITIONS (in natural units) ---
# ... (This section is unchanged) ...
radii_thin_nat = np.random.exponential(scale=L_R_THIN_NAT, size=N_STARS_THIN)
heights_thin_nat = np.random.exponential(scale=L_Z_THIN_NAT, size=N_STARS_THIN) * np.random.choice([-1, 1], size=N_STARS_THIN)
phis_thin = np.random.uniform(0, 2 * np.pi, size=N_STARS_THIN)

radii_thick_nat = np.random.exponential(scale=L_R_THICK_NAT, size=N_STARS_THICK)
heights_thick_nat = np.random.exponential(scale=L_Z_THICK_NAT, size=N_STARS_THICK) * np.random.choice([-1, 1], size=N_STARS_THICK)
phis_thick = np.random.uniform(0, 2 * np.pi, size=N_STARS_THICK)

radii_bulge_nat = np.random.exponential(scale=L_R_BULGE_NAT, size=N_STARS_BULGE)
heights_bulge_nat = np.random.exponential(scale=L_R_BULGE_NAT, size=N_STARS_BULGE) * np.random.choice([-1, 1], size=N_STARS_BULGE)
phis_bulge = np.random.uniform(0, 2 * np.pi, size=N_STARS_BULGE)

radii_halo_nat = np.random.exponential(scale=L_R_HALO_NAT, size=N_STARS_HALO)
heights_halo_nat = np.random.exponential(scale=L_R_HALO_NAT, size=N_STARS_HALO) * np.random.choice([-1, 1], size=N_STARS_HALO)
phis_halo = np.random.uniform(0, 2 * np.pi, size=N_STARS_HALO)


# --- 3. GENERATE VELOCITIES (in natural units) ---
# ... (This section is unchanged and correct) ...
v_circ_thin_nat = np.array([galpy_vcirc(pot_axisymmetric, r) for r in radii_thin_nat])
vR_thin_nat = np.random.normal(0.0, SIGMA_V_THIN_NAT, size=N_STARS_THIN)
vT_thin_nat = v_circ_thin_nat + np.random.normal(0.0, SIGMA_V_THIN_NAT, size=N_STARS_THIN)
vz_thin_nat = np.random.normal(0.0, SIGMA_V_THIN_NAT, size=N_STARS_THIN)

v_circ_thick_nat = np.array([galpy_vcirc(pot_axisymmetric, r) for r in radii_thick_nat])
vR_thick_nat = np.random.normal(0.0, SIGMA_V_THICK_NAT, size=N_STARS_THICK)
vT_thick_nat = v_circ_thick_nat + np.random.normal(0.0, SIGMA_V_THICK_NAT, size=N_STARS_THICK)
vz_thick_nat = np.random.normal(0.0, SIGMA_V_THICK_NAT, size=N_STARS_THICK)

v_circ_bulge_nat = np.array([galpy_vcirc(pot_axisymmetric, r) for r in radii_bulge_nat])
vR_bulge_nat = np.random.normal(0.0, SIGMA_V_BULGE_NAT, size=N_STARS_BULGE)
vT_bulge_nat = v_circ_bulge_nat + np.random.normal(0.0, SIGMA_V_BULGE_NAT, size=N_STARS_BULGE)
vz_bulge_nat = np.random.normal(0.0, SIGMA_V_BULGE_NAT, size=N_STARS_BULGE)

v_circ_halo_nat = np.array([galpy_vcirc(pot_axisymmetric, r) for r in radii_halo_nat])
vR_halo_nat = np.random.normal(0.0, SIGMA_V_HALO_NAT, size=N_STARS_HALO)
vT_halo_nat = v_circ_halo_nat + np.random.normal(0.0, SIGMA_V_HALO_NAT, size=N_STARS_HALO)
vz_halo_nat = np.random.normal(0.0, SIGMA_V_HALO_NAT, size=N_STARS_HALO)

print("Successfully generated initial positions and velocities.")


pot_task5 = MWPotential2014 

# --- 4. RUN STABILITY SIMULATION ---
print("Preparing to run simulation...")

# Combine all stars into single arrays (still in natural units)
all_R_nat = np.concatenate([radii_thin_nat, radii_thick_nat, radii_bulge_nat, radii_halo_nat])
all_vR_nat = np.concatenate([vR_thin_nat, vR_thick_nat, vR_bulge_nat, vR_halo_nat])
all_vT_nat = np.concatenate([vT_thin_nat, vT_thick_nat, vT_bulge_nat, vT_halo_nat])
all_z_nat = np.concatenate([heights_thin_nat, heights_thick_nat, heights_bulge_nat, heights_halo_nat])
all_vz_nat = np.concatenate([vz_thin_nat, vz_thick_nat, vz_bulge_nat, vz_halo_nat])
all_phi = np.concatenate([phis_thin, phis_thick, phis_bulge, phis_halo])

# --- START OF FIX ---
# Stack the natural-unit arrays into the [R, vR, vT, z, vz, phi] format
# that galpy's Orbit initializer expects
vxvv = np.vstack([all_R_nat, all_vR_nat, all_vT_nat, all_z_nat, all_vz_nat, all_phi]).T

# Initialize Orbit from natural units, and provide ro, vo
# This tells galpy how to scale the potentials for the integration
print("Initializing orbits from natural units...")
orbits = Orbit(vxvv, ro=ro, vo=vo)
# --- END OF FIX ---


# Set integration time: 2.5 Gyr (as in your plot)
ts = np.array([0.0, 1]) * u.Gyr

# *** USE THE TASK 5 POTENTIAL FOR INTEGRATION ***
print(f"Integrating orbits using Task 5 potential over {ts[-1]}... This may take several minutes.")
orbits.integrate(ts, pot_task5)
print("Integration complete!")

# --- 5. (TASK 4) REPRODUCE RADIAL VELOCITY MAP ---
print("Task 4/5: Reproducing the Radial Velocity Sky Map")
print("Calculating final coordinates and velocities...")

# Get the final time step
final_time = ts[-1]

# --- START OF FIX ---
# Define the Sun's parameters *now* for the final coordinate transformation
v_sun_total = CartesianDifferential([11.1, vo + 12.24, 7.25] * u.km/u.s)

# Get the final SkyCoord, telling galpy what Sun parameters to use
# This is where the conversion from Galactocentric to Heliocentric happens
print("Converting to Heliocentric coordinates...")
skycoord_final_gc = orbits.SkyCoord(final_time,
                                    galcen_distance=ro * u.kpc,
                                    galcen_v_sun=v_sun_total,
                                    z_sun=0.0 * u.pc)

# Transform to the heliocentric ICRS frame
skycoord_final_icrs = skycoord_final_gc.transform_to('icrs')
# --- END OF FIX ---


# Now get the Galactic coordinates (l, b) and heliocentric radial velocity
b_rad = skycoord_final_icrs.galactic.b.rad
v_rad = skycoord_final_icrs.radial_velocity.to_value(u.km/u.s)

# --- 6. Prepare Data for Plotting ---
l_plot = skycoord_final_icrs.galactic.l.wrap_at(180 * u.deg).rad
print("Data calculated. Now plotting...")

# --- Save data to text file ---
data = np.column_stack([
    skycoord_final_icrs.galactic.l.wrap_at(180 * u.deg).deg,  # l [deg]
    skycoord_final_icrs.galactic.b.deg,                       # b [deg]
    skycoord_final_icrs.radial_velocity.to_value(u.km/u.s)    # v_rad [km/s]
])

np.savetxt("heliocentric_data.txt", data,
           header="l_deg  b_deg  v_rad_km_per_s",
           fmt="%.6f")

print("✅ Data saved to heliocentric_data2.txt")


# --- 7. Create the All-Sky Mean Velocity Plot (like Gaia) ---
# ... (This section is unchanged and correct) ...
print("Creating binned mean-velocity map...")

NBINS_L = 360  # Number of bins in longitude
NBINS_B = 180  # Number of bins in latitude
l_bins = np.linspace(-np.pi, np.pi, NBINS_L)
b_bins = np.linspace(-np.pi/2, np.pi/2, NBINS_B)

mean_vrad_map, _, _, _ = binned_statistic_2d(
    l_plot, b_rad, v_rad,
    statistic='mean',
    bins=[l_bins, b_bins]
)

L_GRID, B_GRID = np.meshgrid(l_bins, b_bins)

plt.figure(figsize=(10, 5))
ax = plt.subplot(111, projection="mollweide")

# Plot the 2D map. cmap='RdBu' matches the PDF caption:
# Red=receding (+v), Blue=approaching (-v)
im = ax.pcolormesh(L_GRID, B_GRID, mean_vrad_map.T,
                   cmap='RdBu', vmin=-200, vmax=200,
                   shading='auto')

cbar = plt.colorbar(im, orientation='horizontal', pad=0.08, shrink=0.7)
cbar.set_label('Mean Heliocentric Radial Velocity (km/s)')

ax.set_title("Simulated All-Sky Mean Velocity Map (with Bar)")
ax.grid(True)
ax.set_xticklabels(['150°', '120°', '90°', '60°', '30°', '0°',
                    '-30°', '-60°', '-90°', '-120°', '-150°'])

plt.show()

print("\nAll tasks complete.")