import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# ==== USER SETTINGS ====
data_dir = "/home/jurij/Documents/Faks/PSUF/nal_1/PSUF_Hmumu/moja_koda/data/generated_histograms/"
n_bins = 100                          # number of bins
x_range = (110, 160)                 # mass range in GeV
save_fig = False                 # change to False if you just want to view
output_dir = "/home/jurij/Documents/Faks/PSUF/nal_1"  # folder to save npz


datasets = {
    "Background": "mc_bkg_new_histogram",
    "Signal": "mc_sig_histogram",
    "Data": "data_histogram",
}

def fit_background_polynomials():
    # ==== FIT FUNCTION (example: exponential + polynomial term) ====
    def background_model1(x, a, b, c):
        """Simple 2nd-degree polynomial background model."""
        return a * np.exp(b * x) + c

    def background_model2(x, a, b, c, d):
        """More complex 3rd-degree polynomial background model."""
        return a * np.exp(b * x) + c * x + d

    def background_model3(x, a, b, c, d, e):
        """Even more complex 4th-degree polynomial background model."""
        return a * np.exp(b * x) + c * x**2 + d * x + e

    bkg_data = np.load(os.path.join(data_dir, "mc_bkg_new_histogram.npz"))

    x_bkg, y_bkg, yerr_bkg = bkg_data["bin_centers"], bkg_data["bin_values"], bkg_data["bin_errors"]

    # ==== PLOT RESULT ====
    plt.figure(figsize=(8, 5))


    for model in [background_model1, background_model2, background_model3]:
        print(f"Model: {model.__name__}, Number of parameters: {model.__code__.co_argcount - 1}")
        # Fit the model to the background data
        # set a different initial guess depending on the model's parameter count
        npar = model.__code__.co_argcount - 1
        if npar == 3:
            # [a, b, c]
            p0 = [y_bkg.max(), -0.02, max(0.0, y_bkg.min())]
        elif npar == 4:
            # [a, b, c, d]
            p0 = [y_bkg.max(), -0.02, 0.1 * y_bkg.max(), max(0.0, y_bkg.min())]
        elif npar == 5:
            # [a, b, c, d, e]
            p0 = [y_bkg.max(), -0.02, 0.01 * y_bkg.max(), 0.1 * y_bkg.max(), max(0.0, y_bkg.min())]
        else:
            # generic fallback
            p0 = [y_bkg.max()] + [-0.01] * (npar - 1)

        print("Initial guess p0:", p0)
        try:
            popt, pcov = curve_fit(model, x_bkg, y_bkg, p0=p0)
            perr = np.sqrt(np.diag(pcov))
            print("Fitted parameters:")
            for name, val, err in zip(["a", "b", "c", "d", "e"][:len(popt)], popt, perr):
                print(f"{name} = {val:.3e} ± {err:.3e}")

            # Plot the fitted model
            x_fit = np.linspace(*x_range, 500)
            y_fit = model(x_fit, *popt)
            degree = max(0, npar - 1)
            plt.plot(x_fit, y_fit, lw=2, label=f"polinom stopnje {degree}")
        except Exception as e:
            print(f"Fit failed: {e}")
        print("-" * 40)




    ## ==== FIT THE BACKGROUND ====
    #p0 = [y_bkg.max(), -0.02, max(0.0, y_bkg.min())]
    #popt, pcov = curve_fit(background_model1, x_bkg, y_bkg, p0=p0)
    #perr = np.sqrt(np.diag(pcov))


    # Data histogram
    #plt.errorbar(x_data, y_data, yerr=yerr_data, fmt="o", color="black", label="Data", markersize=4, capsize=2)
    plt.errorbar(x_bkg, y_bkg, yerr=yerr_bkg, fmt="s", color="tab:blue", label="Simulacija ozadja", markersize=3, alpha=0.7)

    # Background simulation

    plt.xlabel(r"$m_{\mu\mu}$ [GeV]")
    plt.ylabel("Število dogodkov / bin")
    plt.title(f"Fitanje ozadja ({n_bins} binov, območje {x_range[0]}–{x_range[1]} GeV)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(4200, 70000)
    plt.yscale('log')
    plt.tight_layout()

    if save_fig:
        plt.savefig("fit_background.png", dpi=300)
    plt.show()

def fit_background_cms(i):

    # ==== FIT FUNCTION (CMS function) ====
    def cms_fit_function(m, a_1, a_2, a_3):
        m_Z, g_Z = 91.1876, 2.4952
        return np.exp(a_2 * m + a_3 * m**2) / ((m - m_Z) ** a_1 + (0.5 * g_Z) ** a_1)
    
    # Load background and data
    data_types = ["Background", "Data"]
    data = np.load(os.path.join(data_dir, datasets[data_types[i]] + "_histogram.npz"))
    x_bkg, y_bkg, yerr_bkg = data["bin_centers"], data["bin_values"], data["bin_errors"]

    # Fit the model to the background data
    #p0 = np.array([3.288e+00, 2.997e-01, -1.011e-03])
    p0 = np.array([1.0, 1.0, 1.0])*1e-3
    print("Initial guess p0:", p0)
    
    popt, pcov = curve_fit(cms_fit_function, x_bkg, y_bkg, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    print("Fitted parameters:")
    for name, val, err in zip(["a_1", "a_2", "a_3"], popt, perr):
        print(f"{name} = {val:.3e} ± {err:.3e}")

    plt.figure(figsize=(8, 5))
    # Plot the fitted model
    x_fit = np.linspace(*x_range, 500)
    y_fit = cms_fit_function(x_fit, *popt)
    plt.plot(x_fit, y_fit, lw=2, label=f"CMS fit function")
    

    # ==== PLOT RESULT ====

    # Data histogram
    plt.errorbar(x_bkg, y_bkg, yerr=yerr_bkg, fmt="s", color="tab:blue", label="Simulacija ozadja", markersize=3, alpha=0.7)

    # Background simulation

    plt.xlabel(r"$m_{\mu\mu}$ [GeV]")
    plt.ylabel("Število dogodkov / bin")
    plt.title(f"Fitanje ozadja")
    plt.legend()
    plt.grid(alpha=0.3)
    #plt.ylim(3900, 100000)
    plt.yscale('log')
    plt.xlim(110, 160)
    plt.tight_layout()

    if save_fig:
        plt.savefig("fit_background.png", dpi=300)
    plt.show()

def fit_background_cms_exclude(exclude_window=(120, 130)):
    # ==== FIT FUNCTION (CMS function) ====
    def cms_fit_function(m, a_1, a_2, a_3):
        m_Z, g_Z = 91.1876, 2.4952
        return np.exp(a_2 * m + a_3 * m**2) / ((m - m_Z) ** a_1 + (0.5 * g_Z) ** a_1)

    data = np.load(os.path.join(data_dir, "data_histogram.npz"))
    x_all, y_all, yerr_all = data["bin_centers"], data["bin_values"], data["bin_errors"]

    # Create mask to exclude specified window (so signal region is not used in the fit)
    lo, hi = exclude_window
    mask = ~((x_all >= lo) & (x_all <= hi))
    x_fit = x_all[mask]
    y_fit = y_all[mask]
    yerr_fit = yerr_all[mask]

    # Initial guess
    p0 = np.array([1.0, 1.0, 1.0]) * 1e-3
    print("Initial guess p0:", p0)

    try:
        popt, pcov = curve_fit(
            cms_fit_function,
            x_fit,
            y_fit,
            sigma=yerr_fit,
            p0=p0,
            maxfev=10000
        )
        perr = np.sqrt(np.diag(pcov))
        print("Fitted parameters:")
        for name, val, err in zip(["a_1", "a_2", "a_3"], popt, perr):
            print(f"{name} = {val:.3e} ± {err:.3e}")
    except Exception as e:
        print(f"Fit failed: {e}")
        return

    # Plot result
    plt.figure(figsize=(8, 5))
    x_smooth = np.linspace(*x_range, 500)
    y_smooth = cms_fit_function(x_smooth, *popt)
    plt.plot(x_smooth, y_smooth, lw=2, label="CMS fit funkcija (izvzeto {:.0f}-{:.0f} GeV)".format(lo, hi))

    # Show the full histogram (including excluded region) for context
    plt.errorbar(x_all, y_all, yerr=yerr_all, fmt="s", color="tab:blue",
                 label="Simulacija ozadja / podatki", markersize=3, alpha=0.7)

    # Shade the excluded window so it's visible on the plot
    plt.axvspan(lo, hi, color="gray", alpha=0.25, label=f"Izvzeto območje {lo}-{hi} GeV")

    plt.xlabel(r"$m_{\mu\mu}$ [GeV]")
    plt.ylabel("Število dogodkov / bin")
    plt.title("Fitanje ozadja (CMS funkcija) — ekskluzija signala")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.yscale("log")
    plt.xlim(x_range)
    plt.tight_layout()

    if save_fig:
        plt.savefig("fit_background_exclude_window.png", dpi=300)
    plt.show()

def fit_signal_crystalball():

    # ==== FIT FUNCTION (Crystal Ball) ====
    def CrystalBall(x, A, aL, aR, nL, nR, mCB, sCB):
        condlist = [
            (x - mCB) / sCB <= -aL,
            (x - mCB) / sCB >= aR,
        ]
        funclist = [
            lambda x: A
            * (nL / np.abs(aL)) ** nL
            * np.exp(-(aL**2) / 2)
            * (nL / np.abs(aL) - np.abs(aL) - (x - mCB) / sCB) ** (-nL),
            lambda x: A
            * (nR / np.abs(aR)) ** nR
            * np.exp(-(aR**2) / 2)
            * (nR / np.abs(aR) - np.abs(aR) + (x - mCB) / sCB) ** (-nR),
            lambda x: A * np.exp(-((x - mCB) ** 2) / (2 * sCB**2)),
        ]
        return np.piecewise(x, condlist, funclist)

    sign_data = np.load(os.path.join(data_dir, "mc_sig_histogram.npz"))
    x_sign, y_sign, yerr_sign = sign_data["bin_centers"], sign_data["bin_values"], sign_data["bin_errors"]

    popt, pcov = curve_fit(
        CrystalBall,
        x_sign,
        y_sign,
        sigma= yerr_sign,
        p0=[133.0, 1.5, 1.5, 3.7, 9.6, 124.5, 3.0],
        )

    std = np.sqrt(np.diag(pcov))
    fit_values = CrystalBall(x_sign, *popt)
    print("Fitted parameters:")
    for name, val, err in zip(["A", "aL", "aR", "nL", "nR", "mCB", "sCB"], popt, std):
        print(f"{name} = {val:.3e} ± {err:.3e}")
    
    plt.figure(figsize=(8, 5))
    # Plot the fitted model
    x_fit = np.linspace(*x_range, 500)
    y_fit = CrystalBall(x_fit, *popt)
    plt.plot(x_fit, y_fit, lw=2, label=f'Crystal Ball fit')
    # Data histogram
    plt.errorbar(x_sign, y_sign, yerr=yerr_sign, fmt="o", color="tab:orange", label="Simulacija signala", markersize=3, alpha=0.7)
    
    # Background simulation
    plt.xlabel(r"$m_{\mu\mu}$ [GeV]")
    plt.ylabel("Število dogodkov / bin")
    plt.title(f"Fitanje signala")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.yscale('log')
    plt.xlim(110, 160)
    plt.tight_layout()
    if save_fig:
        plt.savefig("fit_signal.png", dpi=300)
    plt.show()

