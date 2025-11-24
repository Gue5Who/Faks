import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import os

# =======================================================
# Nastavitve
# =======================================================

G = 1.0
masses = np.array([1.0, 1.0, 1.0])   # vse mase enake

# Moorejeva "figure-eight" konfiguracija
positions0 = np.array([
    [ 0.97000436, -0.24308753],
    [-0.97000436,  0.24308753],
    [ 0.0,         0.0       ]
])

velocities0 = np.array([
    [ 0.466203685,  0.43236573],
    [ 0.466203685,  0.43236573],
    [-0.93240737 , -0.86473146]
])

# Simulacijski parametri
dt = 0.001
t_end = 20.0
n_steps = int(t_end / dt)

# =======================================================
# Funkcija za pospeške
# =======================================================

def accelerations(pos, masses):
    N = len(masses)
    a = np.zeros_like(pos)
    for i in range(N):
        for j in range(N):
            if i == j: 
                continue
            r = pos[j] - pos[i]
            r2 = np.dot(r, r) + 1e-12
            r3 = r2 * np.sqrt(r2)
            a[i] += G * masses[j] * r / r3
    return a

# =======================================================
# Leapfrog inicializacija
# =======================================================

a0 = accelerations(positions0, masses)
vel_half = velocities0 + 0.5 * dt * a0

positions = np.zeros((n_steps, 3, 2))
positions[0] = positions0

# =======================================================
# Integracija (s progress barom)
# =======================================================

for i in tqdm(range(1, n_steps), desc="Integriram sistem"):
    positions[i] = positions[i-1] + vel_half * dt
    a = accelerations(positions[i], masses)
    vel_half = vel_half + a * dt

# =======================================================
# Statični graf — tirnice (PDF)
# =======================================================

os.makedirs("figs", exist_ok=True)

plt.figure(figsize=(6, 6))
colors = ["tab:blue", "tab:orange", "tab:green"]

for k in range(3):
    x = positions[:, k, 0]
    y = positions[:, k, 1]
    plt.plot(x, y, color=colors[k], label=f"zvezda {k+1}")
    plt.scatter(x[0], y[0], color=colors[k], s=20)

plt.title("Tirnice Moorejeve figure-8 rešitve")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.legend()

pdf_path = "figs/moore_figure8.pdf"
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()
print(f"PDF graf shranjen v: {pdf_path}")


# =======================================================
# VPRAŠANJE ZA ANIMACIJO
# =======================================================

choice = input("Želite ustvariti animacijo? (y/n): ").strip().lower()
make_animation = (choice == "y")

# =======================================================
# ANIMACIJA — samo če upor. izbere 'y'
# =======================================================

if make_animation:

    print("Začenjam izdelavo animacije...")

    fig, ax = plt.subplots(figsize=(6, 6))    # Izračunamo globalne meje
    x_min = np.min(positions[:,:,0])
    x_max = np.max(positions[:,:,0])
    y_min = np.min(positions[:,:,1])
    y_max = np.max(positions[:,:,1])

    # Enake meje osi:
    global_min = min(x_min, y_min)
    global_max = max(x_max, y_max)

    # Dodaj malo "paddinga"
    pad = 0.1 * (global_max - global_min)
    global_min -= pad
    global_max += pad

    # Nastavi enake osi
    ax.set_xlim(global_min, global_max)
    ax.set_ylim(global_min, global_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Animacija Moorejeve figure-8 rešitve")
    ax.grid(True)

    points = [ax.plot([], [], 'o', markersize=6)[0] for _ in range(3)]

    def init():
        for p in points:
            p.set_data([], [])
        return points

    def update(frame):
        for k in range(3):
            points[k].set_data(positions[frame, k, 0], positions[frame, k, 1])
        return points

    # Izločimo preveč sličic — animacija 3× hitrejša
    frames = range(0, n_steps, 30)

    ani = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=True, interval=10
    )

    mp4_path = "figs/moore_figure8_animation.mp4"
    ani.save(mp4_path, writer="ffmpeg", dpi=150)
    plt.close()

    print(f"Animacija shranjena v: {mp4_path}")
else:
    print("Animacija preskočena.")
