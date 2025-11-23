import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter

# =======================================================
# KONSTANTE
# =======================================================

G = 1.0

m1 = 0.5
m2 = 3.0
MBH = 1e6
masses = np.array([m1, m2, MBH])

dt = 0.001
t_end = 5.0
n_steps = int(t_end / dt)

v_0 = 150.0
R0 = 200.0


# =======================================================
# FUNKCIJE ZA SISTEM
# =======================================================

def setup_binary_around_com(a, m1, m2):
    M = m1 + m2
    r1 = a * m2 / M
    r2 = a * m1 / M

    pos1 = np.array([-r1, 0.0])
    pos2 = np.array([ r2, 0.0])

    omega = np.sqrt(G * M / a**3)

    v1 = np.array([0.0, -omega * r1])
    v2 = np.array([0.0,  omega * r2])

    return pos1, pos2, v1, v2


def setup_encounter_falling(a_binary, R0, b, m1, m2, MBH):
    pos1_loc, pos2_loc, v1_loc, v2_loc = setup_binary_around_com(a_binary, m1, m2)

    com_pos = np.array([-R0, b])
    bh_pos  = np.array([0.0, 0.0])

    v_com = np.array([v_0, 0.0])

    positions0 = np.zeros((3, 2))
    velocities0 = np.zeros((3, 2))

    positions0[0] = com_pos + pos1_loc
    velocities0[0] = v_com + v1_loc

    positions0[1] = com_pos + pos2_loc
    velocities0[1] = v_com + v2_loc

    positions0[2] = bh_pos
    velocities0[2] = np.array([0.0, 0.0])

    return positions0, velocities0


def accelerations(positions, masses):
    N = len(masses)
    a = np.zeros_like(positions)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            r = positions[j] - positions[i]
            dist2 = np.dot(r, r) + 1e-12
            dist3 = dist2 * np.sqrt(dist2)
            a[i] += G * masses[j] * r / dist3
    return a


def binary_binding_energy(pos, vel, m1, m2):
    r12 = np.linalg.norm(pos[1] - pos[0])
    v12 = np.linalg.norm(vel[1] - vel[0])
    mu = m1 * m2 / (m1 + m2)
    E = 0.5 * mu * v12**2 - G * m1 * m2 / r12
    return E


# =======================================================
# ENA SAMA SIMULACIJA
# =======================================================

def run_single_simulation(a_binary, b_impact):

    positions0, velocities0 = setup_encounter_falling(
        a_binary=a_binary,
        R0=R0,
        b=b_impact,
        m1=m1,
        m2=m2,
        MBH=MBH
    )

    a0 = accelerations(positions0, masses)
    vel_half = velocities0 + 0.5 * dt * a0

    positions = np.zeros((n_steps, 3, 2))
    positions[0] = positions0

    for i in range(1, n_steps):
        positions[i] = positions[i-1] + vel_half * dt
        a = accelerations(positions[i], masses)
        vel_half = vel_half + a * dt

    # compute final velocities
    a_last = accelerations(positions[-1], masses)
    vel_final = vel_half - 0.5 * dt * a_last

    # return binding energy of binary
    Ebin = binary_binding_energy(
        positions[-1][:2],
        vel_final[:2],
        m1, m2
    )

    return Ebin

def run_single_simulation_with_series(a_binary, b_impact):
    """
    Ista simulacija kot run_single_simulation,
    ampak shrani in vrne celotno časovno serijo vezavne energije.
    """
    positions0, velocities0 = setup_encounter_falling(
        a_binary=a_binary,
        R0=R0,
        b=b_impact,
        m1=m1,
        m2=m2,
        MBH=MBH
    )

    # inicializacija
    a0 = accelerations(positions0, masses)
    vel_half = velocities0 + 0.5 * dt * a0

    positions = np.zeros((n_steps, 3, 2))
    velocities = np.zeros((n_steps, 3, 2))

    positions[0] = positions0
    velocities[0] = velocities0

    # integracija
    for i in range(1, n_steps):
        positions[i] = positions[i-1] + vel_half * dt
        a = accelerations(positions[i], masses)
        vel_half = vel_half + a * dt
        velocities[i] = vel_half - 0.5 * dt * a  # fizikalne hitrosti

    # izračun celotne serije vezavne energije
    E_series = []
    for i in range(n_steps):
        pos_bin = positions[i][:2]
        vel_bin = velocities[i][:2]
        E = binary_binding_energy(pos_bin, vel_bin, m1, m2)
        E_series.append(E)

    return np.array(E_series)



def binding_energy_timeseries(positions_timeseries, velocities_timeseries, m1, m2):
    """
    Vrne seznam vezavnih energij dvozvezdja E_bin(t) skozi celotno simulacijo.
    """
    E_series = []
    for t in range(len(positions_timeseries)):
        pos = positions_timeseries[t][:2]
        vel = velocities_timeseries[t][:2]
        E = binary_binding_energy(pos, vel, m1, m2)
        E_series.append(E)
    return np.array(E_series)


# =======================================================
# ŠTUDIJA PARAMETROV (GLAVNI DEL NALOGE)
# =======================================================

def study_parameter_space():

    a_values = [0.5, 1.0, 2.0, 3.0]
    b_values = [10, 20, 30, 40, 50]

    results = []

    for a in a_values:
        for b in b_values:

            print(f"\n==============================")
            print(f" Simulacija za a={a}, b={b}")
            print(f"==============================")

            Ebin = run_single_simulation(a_binary=a, b_impact=b)

            status = "VEZANO" if Ebin < 0 else "RAZPAD"

            print(f"Končna vezavna energija: {Ebin:.5f}   -> {status}")

            results.append((a, b, Ebin))

    print("\n\n=== KONČNI REZULTATI ===")
    for (a, b, E) in results:
        status = "VEZANO" if E < 0 else "RAZPAD"
        print(f"a={a:4}   b={b:4}   Ebin={E:10.4f}   {status}")

# =======================================================
# GLAVNI ZAGON
# =======================================================

def plot_binding_energy(E_series, dt, a_binary, b_impact):
    t = np.arange(len(E_series)) * dt

    plt.figure(figsize=(8,5))
    plt.plot(t, E_series, linewidth=1.5)

    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("čas")
    plt.ylabel("vezavna energija E_bin")
    plt.title(f"Vezavna energija dvozvezdja skozi čas (a={a_binary}, b={b_impact})")
    plt.grid(True)

    # shrani
    os.makedirs("figs/en", exist_ok=True)
    fname = f"figs/en/vezavna_energija_a{a_binary}_b{b_impact}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

    print(f"Graf vezavne energije shranjen v: {fname}")



def run_and_plot(a, b):
    E_series = run_single_simulation_with_series(a, b)
    plot_binding_energy(E_series, dt, a, b)


def scan_parameter_space_simple():

    # izberi več vrednosti a in b
    a_values = np.linspace(0.01, 4.0, 50)
    b_values = np.linspace(0.01, 140.0, 100)

    results = []
    a_vez = []
    b_vez = []
    vezano_only = a_vez, b_vez
    print("\n=== Začenjam preprost parameter scan (a,b) ===\n")

    for a in a_values:
        for b in b_values:

            print(f"\n==============================")
            print(f" Simulacija za a={a:.3f}, b={b:.3f}")
            print(f"==============================")

            Ebin = run_single_simulation(a_binary=a, b_impact=b)

            status = "VEZANO" if Ebin < 0 else "RAZPAD"

            print(f"Končna vezavna energija: {Ebin:.5f}   -> {status}")

            results.append((a, b, Ebin))

            # če je VEZANO, shrani parametre
            if Ebin < 0:
                a_vez.append(a)
                b_vez.append(b)

    # tisk končnih rezultatov
    print("\n\n=== KONČNI REZULTATI ===")
    for (a, b, E) in results:
        status = "VEZANO" if E < 0 else "RAZPAD"
        print(f"a={a:6.3f}   b={b:6.3f}   Ebin={E:10.4f}   {status}")

    return a_vez, b_vez


def plot_vezane():
    a, b = scan_parameter_space_simple()

    plt.figure(figsize=(8,6))

    plt.scatter(
        a, b,
        s=10,
        c=b,              # barvni gradient po parametru b
        cmap="plasma",
        edgecolors="black",
        linewidths=0.4,
        alpha=0.9
    )

    plt.xlabel("razdalja med zvezdama a", fontsize=12)
    plt.ylabel("vpadni parameter b", fontsize=12)
    plt.title("Pare (a, b), za katere dvozvezdje ostane VEZANO", fontsize=14)

    plt.grid(alpha=0.3)
    cbar = plt.colorbar()
    cbar.set_label("vpadni parameter b")

    plt.tight_layout()

    # Save PDF
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/vezani_scatter.pdf", bbox_inches="tight")

    plt.show()







#if __name__ == "__main__":
    
    #study_parameter_space()
    #for i in range(len(a_values)):
    #    for j in range(len(b_values)):
    #        run_and_plot(a_values[i], b_values[j])

plot_vezane()