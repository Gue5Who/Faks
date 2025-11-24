import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
from collections import Counter
import pandas as pd

#path_spektri = '/home/jurijs/Documents/Faks/PSUF/nal_2/spektri/'
path_spektri = '/home/jurij/Documents/Faks/PSUF/nal_2/spektri/'
path = '/home/jurij/Documents/Faks/PSUF/nal_2/'
#path = '/home/jurijs/Documents/Faks/PSUF/nal_2/'

spec_ids = np.arange(1,10000,1)
spectra = [np.loadtxt(path_spektri + f"{sid}.dat", comments="#") for sid in spec_ids]
X = np.array(spectra)


# Compute distance from mean for all spectra (assuming X shape = [n, p])
mean_spec = np.mean(X, axis=0)
distances = np.linalg.norm(X - mean_spec, axis=1)

# Load training labels (MAB, BIN, etc.)
type_data = np.loadtxt(path + "learning_set_types.txt", dtype=str)
type_ids = type_data[:, 0].astype(int)
type_labels = type_data[:, 1]

# Match labels to loaded spectra
labels = []
for sid in spec_ids:
    if sid in type_ids:
        idx = np.where(type_ids == sid)[0][0]
        labels.append(type_labels[idx])
    else:
        labels.append("UNKNOWN")
labels = np.array(labels)

# Color map
color_map = {
    "MAB": "tab:green",
    "BIN": "tab:blue",
    "TRI": "tab:cyan",
    "HFR": "tab:orange",
    "HAE": "tab:red",
    "CMP": "tab:purple",
    "DIB": "tab:brown",
    "UNKNOWN": "lightgray"
}
colors = [color_map.get(lbl, "gray") for lbl in labels]


# IDENTIFY OUTLIER SPECTRA

mean_spec = np.mean(X, axis=0)
distances = [euclidean(spec, mean_spec) for spec in X]

# Try a few perplexity values to see the effect

perplexity = 13

tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    learning_rate=400,
    n_iter=2000,
    init='random',
    random_state=42,
    verbose=1
)

X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(7,6))
mask_unknown = labels == "UNKNOWN"
if np.any(mask_unknown):
    plt.scatter(X_tsne[mask_unknown, 0], X_tsne[mask_unknown, 1],
                label="Neznani tip",
                c="lightgray",
                s=30,
                alpha=0.5,
                edgecolors='k',
                zorder=1)

# Plot known star classes on top
for lbl in np.unique(labels):
    if lbl == "UNKNOWN":
        continue
    mask = labels == lbl
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                label=lbl,
                c=color_map.get(lbl, "gray"),
                s=50,
                edgecolors='k',
                alpha=0.9,
                zorder=2)

# Plot outliers on top of everything
#plt.scatter(X_tsne[outlier_idx, 0], X_tsne[outlier_idx, 1],
#            color='gold',
#            s=100,
#            marker='*',
#            edgecolors='k',
#            linewidth=0.8,
#            label='Odstopanja',
#            zorder=3)

plt.title(f"t-SNE projekcija (perplexity = {perplexity})")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()