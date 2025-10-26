import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tables

# ==== USER SETTINGS ====
data_dir = "data/raw_data/"          # folder where .h5 files are stored
n_bins = 100                          # number of bins
x_range = (110, 160)                 # mass range in GeV
save_hist = True                    # change to False if you just want to view
output_dir = "data/generated_histograms/"  # folder to save npz


# ==== LOAD DATA ====
def load_data(path):
    """Load .h5 dataset."""
    print(f"Loading {path} ...")
    return pd.read_hdf(path, key="ntuple")
    
def make_histogram(data, n_bins=n_bins, x_range=x_range):
    """Return bin centers, values, and errors for given dataset."""
    events = data["Muons_Minv_MuMu_Paper"]
    weights = data["CombWeight"]
    weights2 = weights ** 2

    values, edges = np.histogram(events, bins=n_bins, range=x_range, weights=weights)
    sum_w2, _ = np.histogram(events, bins=n_bins, range=x_range, weights=weights2)
    errors = np.sqrt(sum_w2)
    centers = 0.5 * (edges[1:] + edges[:-1])

    return centers, values, errors
# ==== PLOTTING ====
datasets = {
    "Background": "mc_bkg_new",
    "Signal": "mc_sig",
    "Data": "data",
}

plt.figure(figsize=(8, 5))
colors = {"Background": "tab:blue", "Signal": "tab:orange", "Data": "tab:green"}

for label, name in datasets.items():
    data = load_data(os.path.join(data_dir, name + ".h5"))
    centers, values, errors = make_histogram(data, n_bins, x_range)

    plt.errorbar(centers, values, yerr=errors, label=label, color=colors[label],
                 markersize=1, capsize=2, alpha=0.8, fmt ='o-')
 
    plt.hist(centers, bins=n_bins, range=x_range, weights=values, histtype='step', color=colors[label], alpha=1, label=label)
    
    #os.makedirs(output_dir, exist_ok=True)
    #save_path = os.path.join(output_dir, f"{name}_histogram.npz")
    #np.savez(save_path, bin_edges=centers, bin_centers=centers, bin_values=values, bin_errors=errors)
    #print(f"Saved histogram to {save_path}")


plt.xlabel(r"$m_{\mu\mu}$ [GeV]")
plt.ylabel("Število dogodkov / bin")
plt.title(f"Histogrami ({n_bins} binov)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.yscale('log')

# add measurements/background subplot under the main axes
fig = plt.gcf()
ax_main = plt.gca()
pos = ax_main.get_position()

# shrink main plot to make room for the ratio plot
main_pos = [pos.x0, pos.y0 + 0.22 * pos.height, pos.width, pos.height * 0.78]
ax_main.set_position(main_pos)

# create ratio (measurements/background) axes below, sharing x-axis
ax_ratio = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height * 0.22], sharex=ax_main)

# load histograms for Data (measurements) and Background
bkg_data = load_data(os.path.join(data_dir, datasets["Background"] + ".h5"))
meas_data = load_data(os.path.join(data_dir, datasets["Data"] + ".h5"))
_, bkg_vals, bkg_err = make_histogram(bkg_data, n_bins, x_range)
_, meas_vals, meas_err = make_histogram(meas_data, n_bins, x_range)

# compute Data / Background and propagated uncertainty
eps = 1e-12
mask = bkg_vals > 0
meas_over_bkg = np.zeros_like(meas_vals, dtype=float)
meas_over_bkg_err = np.zeros_like(meas_vals, dtype=float)

meas_over_bkg[mask] = meas_vals[mask] / bkg_vals[mask]
# propagate relative errors, guard against zero measured value
rel_meas = np.where(meas_vals > 0, meas_err / np.maximum(meas_vals, eps), 0.0)
rel_bkg = bkg_err / np.maximum(bkg_vals, eps)
meas_over_bkg_err[mask] = meas_over_bkg[mask] * np.sqrt(rel_meas[mask] ** 2 + rel_bkg[mask] ** 2)

# plot Data/Background
ax_ratio.errorbar(centers, meas_over_bkg, yerr=meas_over_bkg_err, fmt='o-', color='black', markersize=3, capsize=2)
ax_ratio.axhline(1.0, color='gray', linestyle='--', linewidth=0.7)

ax_ratio.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
ax_ratio.set_ylabel("Podatki / Ozadje")
ax_ratio.grid(alpha=0.3)

# tidy x-axis labels (show only on ratio)
plt.setp(ax_main.get_xticklabels(), visible=False)

plt.show()