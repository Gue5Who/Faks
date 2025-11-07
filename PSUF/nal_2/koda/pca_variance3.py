import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

path_spektri = '/home/jurij/Documents/Faks/PSUF/nal_2/spektri/'

path = '/home/jurij/Documents/Faks/PSUF/nal_2/'


# ====================================
# LOAD YOUR DATA (example)
# ====================================
spec_ids = np.arange(1,10000,1)
spectra = [np.loadtxt(path_spektri + f"{sid}.dat", comments="#") for sid in spec_ids]
X = np.array(spectra)  # shape: (n_spectra, n_wavelengths)

# Load wavelengths (for reference)
wav = np.loadtxt(path_spektri + "val.dat", comments="#")

# Load training set parameters (if available)
# Columns: spectrum_id, Teff, logg, [M/H]
param_data = np.loadtxt(path + "training_set_parameters.txt", comments="#")
spec_param_id = param_data[:, 0].astype(int)
Teff = param_data[:, 1]
logg = param_data[:, 2]
MH = param_data[:, 3]

# Match parameter values to the loaded spectra (if IDs overlap)
Teff_match, logg_match, MH_match = [], [], []
for sid in spec_ids:
    if sid in spec_param_id:
        idx = np.where(spec_param_id == sid)[0][0]
        Teff_match.append(Teff[idx])
        logg_match.append(logg[idx])
        MH_match.append(MH[idx])
    else:
        Teff_match.append(np.nan)
        logg_match.append(np.nan)
        MH_match.append(np.nan)

Teff_match = np.array(Teff_match)
logg_match = np.array(logg_match)
MH_match = np.array(MH_match)


# IDENTIFY OUTLIERS (FROM STEP 1)

mean_spec = np.mean(X, axis=0)
distances = [euclidean(spec, mean_spec) for spec in X]
outlier_idx = np.argsort(distances)[-10:]  # 2 most deviant
print("Outlier spectra:", [spec_ids[i] for i in outlier_idx])


# PCA COMPUTATION (from previous step)

# Center data
B = X - np.mean(X, axis=0)
C = np.dot(B.T, B) / (X.shape[0] - 1)
eigvals, eigvecs = np.linalg.eig(C)

# Sort descending
idx = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

# Project onto first 3 components
W = eigvecs[:, :3]
T = np.dot(B, W)


# PLOT: FIRST 2 PCA COMPONENTS

plt.figure(figsize=(7,6))
plt.scatter(T[:,0], T[:,1], c='lightgray', label='Spectra')
plt.scatter(T[outlier_idx,0], T[outlier_idx,1], 
            color='red', s=100, edgecolors='k', label='Outliers')
for i in outlier_idx:
    plt.text(T[i,0], T[i,1], str(spec_ids[i]), fontsize=10, weight='bold')

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Projection onto first two PCA components')
plt.legend()
plt.tight_layout()
plt.show()


# OPTIONAL: COLOR BY PHYSICAL PARAMETER

if not np.isnan(Teff_match).all():
    plt.figure(figsize=(7,6))
    sc = plt.scatter(T[:,0], T[:,1], c=Teff_match, cmap='plasma', s=80)
    plt.colorbar(sc, label='Teff [K]')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA projection colored by effective temperature')
    plt.tight_layout()
    plt.show()

if not np.isnan(MH_match).all():
    plt.figure(figsize=(7,6))
    sc = plt.scatter(T[:,0], T[:,1], c=MH_match, cmap='coolwarm', s=80)
    plt.colorbar(sc, label='[M/H]')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA projection colored by metallicity')
    plt.tight_layout()
    plt.show()

if not np.isnan(logg_match).all():
    plt.figure(figsize=(7,6))
    sc = plt.scatter(T[:,0], T[:,1], c=logg_match, cmap='viridis', s=80)
    plt.colorbar(sc, label='log(g)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA projection colored by surface gravity')
    plt.tight_layout()
    plt.show()
