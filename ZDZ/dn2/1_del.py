import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from tqdm import tqdm
from matplotlib.animation import FFMpegWriter

# ===========================
# Osnovne nastavitve problema
# ===========================

G = 1.0
masses = np.array([1.0, 1.0, 1.0])

# Nepravilni trikotnik (poljubna postavitev)
positions0 = np.array([
    [-1.0,  0.2],
    [ 0.8, -0.5],
    [ 0.3,  0.9]
])

velocities0 = np.zeros((3, 2))

dt = 0.001
t_end = 3.0
n_steps = int(t_end / dt)

# ===========================
# Pomožne funkcije
# ===========================

def accelerations(positions, masses):
    N = len(masses)
    a = np.zeros_like(positions)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            r_ij = positions[j] - positions[i]
            dist2 = np.dot(r_ij, r_ij) + 1e-10
            dist3 = dist2 * np.sqrt(dist2)
            a[i] += G * masses[j] * r_ij / dist3
    return a

# ===========================
# Inicializacija Leapfrog sheme
# ===========================

a0 = accelerations(positions0, masses)
vel_half = velocities0 + 0.5 * dt * a0

positions = np.zeros((n_steps, 3, 2))
positions[0] = positions0
velocities_half = vel_half.copy()

# ===========================
# Glavna integracijska zanka
# ===========================

for i in tqdm(range(1, n_steps), desc="Simulating"):
    positions[i] = positions[i-1] + velocities_half * dt
    a = accelerations(positions[i], masses)
    velocities_half = velocities_half + a * dt

# ==========================================
# Risanje statičnega grafa (PDF)
# ==========================================

os.makedirs("figs", exist_ok=True)

plt.figure(figsize=(6, 6))
colors = ["tab:blue", "tab:orange", "tab:green"]

for k in range(3):
    x = positions[:, k, 0]
    y = positions[:, k, 1]
    plt.plot(x, y, label=f"zvezda {k+1}")
    plt.scatter(x[0], y[0], marker="o")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Tirnice trozvezdnega sistema v nepravilnem trikotniku")
plt.legend()
plt.axis("equal")
plt.grid(True)

pdf_path = "figs/three_body_triangle.pdf"
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()
print(f"Saved static plot to {pdf_path}")

# ==========================================
# ANIMACIJA
# ==========================================


mp4_path = "figs/three_body_triangle_animation.mp4"
writer = FFMpegWriter(fps=60)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(np.min(positions[:,:,0]) - 0.2, np.max(positions[:,:,0]) + 0.2)
ax.set_ylim(np.min(positions[:,:,1]) - 0.2, np.max(positions[:,:,1]) + 0.2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Animacija gibanja trozvezdnega sistema v nepravilnem trikotniku")
ax.grid(True)

points = [ax.plot([], [], 'o', markersize=6)[0] for _ in range(3)]

def draw_frame(i):
    for k in range(3):
        points[k].set_data(positions[i, k, 0], positions[i, k, 1])

# Use fewer frames for speed
frames = range(0, n_steps, 30)

with writer.saving(fig, mp4_path, dpi=150):
    for i in tqdm(frames, desc="Rendering animation"):
        draw_frame(i)
        writer.grab_frame()

plt.close()
print(f"Saved animation to {mp4_path}")