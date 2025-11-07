import numpy as np
import matplotlib.pyplot as plt

path_spektri = '/home/jurij/Documents/Faks/PSUF/nal_2/spektri/'
# ====================================
# LOAD YOUR DATA (example)
# ====================================
spec_ids = np.arange(1,10000,1)
spectra = [np.loadtxt(path_spektri + f"{sid}.dat", comments="#") for sid in spec_ids]
X = np.array(spectra)  # shape: (n_spectra, n_wavelengths)

# ====================================
# CENTER DATA
# ====================================
mean_spectrum = np.mean(X, axis=0)
B = X - mean_spectrum

# ====================================
# COVARIANCE MATRIX
# ====================================
n_samples = X.shape[0]
C = np.dot(B.T, B) / (n_samples - 1)

# ====================================
# EIGENDECOMPOSITION
# ====================================
eigvals, eigvecs = np.linalg.eig(C)

# Sort eigenvalues (and vectors) descending
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# ====================================
# ARIANCE EXPLAINED
# ====================================
variance_ratio = eigvals / np.sum(eigvals)
cumulative_variance = np.cumsum(variance_ratio)

# Automatically find number of components covering >= 90% variance
threshold = 0.90
q_opt = np.argmax(cumulative_variance >= threshold) + 1
print(f"Number of components for >=90% variance: {q_opt}")

# ====================================
# ISUALIZE VARIANCE
# ====================================
plt.figure(figsize=(6,4))
plt.plot(cumulative_variance, 'o-', color='tab:blue')
plt.axhline(y=threshold, color='r', linestyle='--', label='90% threshold')
plt.axvline(x=q_opt, color='g', linestyle='--', label=f'q = {q_opt}')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('Variance explained by PCA components')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ====================================
# 7️⃣ PROJECT ONTO FIRST q_opt COMPONENTS
# ====================================
W = eigvecs[:, :q_opt]   # projection matrix (p × q_opt)
T = np.dot(B, W)         # projected data (n × q_opt)
print(f"Projected data shape: {T.shape}")

# ====================================
# 8️⃣ OPTIONAL 2D VISUALIZATION
# ====================================
if q_opt >= 2:
    plt.figure(figsize=(6,5))
    plt.scatter(T[:,0], T[:,1], c='tab:purple')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'Projection onto first 2 PCA components (q_opt = {q_opt})')
    plt.tight_layout()
    plt.show()