import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

path_spektri = '/home/jurij/Documents/Faks/PSUF/nal_2/spektri/'

path = '/home/jurij/Documents/Faks/PSUF/nal_2/'


# LOAD YOUR DATA

spec_ids = np.arange(1,10000,1)
spectra = [np.loadtxt(path_spektri + f"{sid}.dat", comments="#") for sid in spec_ids]
X = np.array(spectra)  # shape: (n_spectra, n_wavelengths)


# Load training parameters (Teff, logg, [M/H])
param_data = np.loadtxt(path + "training_set_parameters.txt", comments="#")
spec_param_id = param_data[:, 0].astype(int)
Teff = param_data[:, 1]
logg = param_data[:, 2]
MH = param_data[:, 3]

# Match available parameters to loaded spectra
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


# PCA COMPUTATION

# Center data
B = X - np.mean(X, axis=0)
C = np.dot(B.T, B) / (X.shape[0] - 1)

# Eigen decomposition
eigvals, eigvecs = np.linalg.eig(C)

# Sort eigenvalues descending
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]


# VARIANCE ANALYSIS

variance_ratio = eigvals / np.sum(eigvals)
cumulative_variance = np.cumsum(variance_ratio)

# Determine number of components for ≥90% variance
threshold = 0.90
q_opt = np.argmax(cumulative_variance >= threshold) + 1
print(f"Number of components explaining ≥90% variance: {q_opt}")

# Plot explained variance curve
plt.figure(figsize=(7,5))
plt.plot(np.arange(1, len(variance_ratio)+1), cumulative_variance, 'o-', color='tab:blue')
plt.axhline(threshold, color='r', linestyle='--', label='90% threshold')
plt.axvline(q_opt, color='g', linestyle='--', label=f'q = {q_opt}')
plt.xlabel('Number of PCA components')
plt.ylabel('Cumulative explained variance')
plt.title('Cumulative variance explained by PCA components')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# PROJECT DATA AND CHECK PHYSICAL TRENDS

# Project onto the first few components
W = eigvecs[:, :3]
T = np.dot(B, W)

# Plot the first PCA component vs. Teff, [M/H], log g
if not np.isnan(Teff_match).all():
    plt.figure(figsize=(6,4))
    plt.scatter(Teff_match, T[:,0], color='tab:blue')
    plt.xlabel('Teff [K]')
    plt.ylabel('PCA Component 1')
    plt.title('Correlation between PCA1 and Teff')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if not np.isnan(MH_match).all():
    plt.figure(figsize=(6,4))
    plt.scatter(MH_match, T[:,0], color='tab:orange')
    plt.xlabel('[M/H]')
    plt.ylabel('PCA Component 1')
    plt.title('Correlation between PCA1 and metallicity')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if not np.isnan(logg_match).all():
    plt.figure(figsize=(6,4))
    plt.scatter(logg_match, T[:,0], color='tab:green')
    plt.xlabel('log(g)')
    plt.ylabel('PCA Component 1')
    plt.title('Correlation between PCA1 and surface gravity')
    plt.grid(True)
    plt.tight_layout()
    plt.show()