import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# =======================================================
# 3. NALOGA: dvozvezdje + crna luknja
# =======================================================

# Konstantne (v kodnih enotah)
G = 1.0

# Mase (v "solnih masah" - a so tukaj le razmerja)
m1 = 0.5       # prva zvezda
m2 = 3.0       # druga zvezda
MBH = 1e6      # crna luknja
masses = np.array([m1, m2, MBH])

# -------------------------------------------------------
# Nastavitve dvozvezdja in srecanja
# -------------------------------------------------------

# Medsebojna razdalja zvezd v dvozvezdju
a_binary = 1.0          # lahko spreminjas (manjsa = bolj tesno dvozvezdje)

# Zacetna oddaljenost tezisca dvozvezdja od crne luknje
R0 = 30.0               # "zunanja" razdalja

# Vpadni parameter b (koliko "mimo" gre COM)
b_impact = 10.0          # preizkusi razlicne vrednosti
v_0 = 150

# Tip orbite dvozvezdja okoli BH: "parabolic" ali "hyperbolic"
orbit_type = "parabolic"   # ali "hyperbolic"

# Faktor za hiperbolicno orbito (ce orbit_type == "hyperbolic")
hyper_factor = 1.5      # v_inf ~ hyper_factor * v_parabolic


# -------------------------------------------------------
# Casovna integracija
# -------------------------------------------------------

dt = 0.001
t_end = 3.4
n_steps = int(t_end / dt)


# =======================================================
# Funkcije za inicializacijo sistema
# =======================================================

def setup_binary_around_com(a, m1, m2):
    """
    Postavi dvozvezdje v krozno orbito okoli tezisca.
    Vrne lokalne pozicije (relativno na COM) in hitrosti.
    """
    M = m1 + m2

    # Razdalji od COM
    r1 = a * m2 / M
    r2 = a * m1 / M

    # Položaji v smeri x, COM v (0, 0)
    pos1 = np.array([-r1, 0.0])
    pos2 = np.array([ r2, 0.0])

    # Kotna hitrost za krožno orbito (G*M/a^3)^1/2
    omega = np.sqrt(G * M / a**3)

    # Hitrosti v smeri +y/-y
    v1 = np.array([0.0, -omega * r1])
    v2 = np.array([0.0,  omega * r2])

    return pos1, pos2, v1, v2

def setup_encounter_falling(a_binary, R0, b, m1, m2, MBH):
    """
    A proper setup for a distant binary falling toward a supermassive black hole.

    - binary COM placed at (-R0, b)
    - COM velocity = 0 (binary falls inward)
    - binary itself is placed in circular orbit around its COM
    """

    # --- (1) binary local configuration ---
    pos1_loc, pos2_loc, v1_loc, v2_loc = setup_binary_around_com(a_binary, m1, m2)

    # --- (2) place COM far away from BH ---
    com_pos = np.array([-R0, b])
    bh_pos  = np.array([0.0, 0.0])

    # COM velocity = 0 (falling inward)
    v_com = np.array([v_0, 0.0])

    # --- (3) full system initial conditions ---
    positions0 = np.zeros((3, 2))
    velocities0 = np.zeros((3, 2))

    # star 1
    positions0[0] = com_pos + pos1_loc
    velocities0[0] = v_com + v1_loc

    # star 2
    positions0[1] = com_pos + pos2_loc
    velocities0[1] = v_com + v2_loc

    # black hole
    positions0[2] = bh_pos
    velocities0[2] = np.array([0.0, 0.0])

    return positions0, velocities0



# =======================================================
# Gravitacijski pospeski
# =======================================================

def accelerations(positions, masses):
    """
    Izracun gravitacijskih pospeskov za vsa telesa.
    positions: (N, 2)
    masses: (N,)
    """
    N = len(masses)
    a = np.zeros_like(positions)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            r_ij = positions[j] - positions[i]
            dist2 = np.dot(r_ij, r_ij) + 1e-12
            dist3 = dist2 * np.sqrt(dist2)
            a[i] += G * masses[j] * r_ij / dist3
    return a


# =======================================================
# Inicializacija sistema za 3. nalogo
# =======================================================

positions0, velocities0 = setup_encounter_falling(
    a_binary=a_binary,
    R0=200.0,      # start 200 code units away
    b=b_impact,    # grazing distance
    m1=m1,
    m2=m2,
    MBH=MBH
)

# Leapfrog: najprej izracunamo pospesek in half-step hitrosti
a0 = accelerations(positions0, masses)
vel_half = velocities0 + 0.5 * dt * a0

positions = np.zeros((n_steps, 3, 2))
positions[0] = positions0


# =======================================================
# Integracija (Leapfrog) s progress barom
# =======================================================

for i in tqdm(range(1, n_steps), desc="Integriram sistem (3. naloga)"):
    # posodobimo polozaje
    positions[i] = positions[i-1] + vel_half * dt
    # posodobimo pospeske
    a = accelerations(positions[i], masses)
    # posodobimo hitrosti v half-stepu
    vel_half = vel_half + a * dt

# Izracunamo "priblizne" koncne hitrosti v integer casu t_end
a_last = accelerations(positions[-1], masses)
vel_final = vel_half - 0.5 * dt * a_last   # v(t_end)


# =======================================================
# Statični graf tirnic (PDF)
# =======================================================

os.makedirs("figs", exist_ok=True)

plt.figure(figsize=(7, 7))

colors = ["tab:blue", "tab:orange", "black"]
labels = ["zvezda 1 (0.5 M☉)", "zvezda 2 (3 M☉)", "črna luknja (10⁶ M☉)"]

for k in range(3):
    x = positions[:, k, 0]
    y = positions[:, k, 1]
    if k < 2:
        plt.plot(x, y, color=colors[k], label=labels[k])
        plt.scatter(x[0], y[0], color=colors[k], s=15)  # začetni položaj
    else:
        plt.scatter(x[0], y[0], color=colors[k], s=50, marker="x", label=labels[k])

plt.xlabel("x")
plt.ylabel("y")
plt.title("Srečanje dvozvezdja s črno luknjo (3. naloga)")
plt.grid(True)

# Enake meje osi
x_min = np.min(positions[:, :, 0])
x_max = np.max(positions[:, :, 0])
y_min = np.min(positions[:, :, 1])
y_max = np.max(positions[:, :, 1])

global_min = min(x_min, y_min)
global_max = max(x_max, y_max)
pad = 0.1 * (global_max - global_min)
global_min -= pad
global_max += pad

plt.xlim(global_min, global_max)
plt.ylim(global_min, global_max)
plt.gca().set_aspect('equal', adjustable='box')

plt.legend()
pdf_path = "figs/three_body_bh_encounter.pdf"
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print(f"PDF graf shranjen v: {pdf_path}")

# Izpis končnih hitrosti
speed1 = np.linalg.norm(vel_final[0])
speed2 = np.linalg.norm(vel_final[1])
speed_bh = np.linalg.norm(vel_final[2])

print(f"Končna hitrost zvezde 1: {speed1:.5f} (kodne enote)")
print(f"Končna hitrost zvezde 2: {speed2:.5f} (kodne enote)")
print(f"Hitrost črne luknje (pričakovano skoraj 0): {speed_bh:.5e}")


# =======================================================
# Vprasanje za animacijo
# =======================================================

#choice = input("Želite ustvariti animacijo srečanja? (y/n): ").strip().lower()
#make_animation = (choice == "y")

make_animation = True
# =======================================================
# Animacija (s kvadratnimi mejami osi)
# =======================================================

#    if make_animation:
#        print("Začenjam izdelavo animacije...")
#
#        fig, ax = plt.subplots(figsize=(7, 7))
#
#        # Uporabimo enake meje kot zgoraj
#        ax.set_xlim(-200, 200)
#        ax.set_ylim(-200, 200)
#        ax.set_aspect('equal', adjustable='box')

# =======================================================
# DYNAMIC CAMERA ANIMATION
# =======================================================

if make_animation:
    print("Začenjam izdelavo animacije (dinamična kamera)...")

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Animacija srečanja dvozvezdja s črno luknjo")
    ax.grid(True)

    points = [
        ax.plot([], [], 'o', markersize=6, color="tab:blue")[0],
        ax.plot([], [], 'o', markersize=6, color="tab:orange")[0],
        ax.plot([], [], 'x', markersize=8, color="black")[0]
    ]

    def init():
        for p in points:
            p.set_data([], [])
        return points

    def update(frame):
        # Update positions of stars and BH
        for k, p in enumerate(points):
            p.set_data(positions[frame, k, 0], positions[frame, k, 1])

        # ---- Dynamic camera scaling ----
        x = positions[frame, :, 0]
        y = positions[frame, :, 1]

        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)

        # Compute center and half-span
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)

        span = max(xmax - xmin, ymax - ymin)
        span *= 1.3     # padding factor

        # Set equal limits around the center
        ax.set_xlim(cx - span/2, cx + span/2)
        ax.set_ylim(cy - span/2, cy + span/2)
        ax.set_aspect('equal', adjustable='box')

        return points

    # Render fewer frames for speed
    frames = range(0, n_steps, 10)

    ani = animation.FuncAnimation(
        fig, update,
        frames=frames, init_func=init,
        blit=True, interval=10
    )

    mp4_path = "figs/three_body_bh_dynamic_camera_blizu_prelet.mp4"
    ani.save(mp4_path, writer="ffmpeg", dpi=150)
    plt.close()

    print(f"Animacija shranjena v: {mp4_path}")

else:
    print("Animacija preskočena.")

