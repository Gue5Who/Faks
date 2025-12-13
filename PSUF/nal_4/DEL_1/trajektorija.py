import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# -------------------------------------------------------
# Lorenzov sistem (enako kot v tvoji nalogi)
# -------------------------------------------------------
def lorenz63_rhs(state, sigma=10, r=28, b=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (r - z) - y
    dz = x * y - b * z
    return np.array([dx, dy, dz])

def rk4_step(state, dt):
    k1 = lorenz63_rhs(state)
    k2 = lorenz63_rhs(state + 0.5*dt*k1)
    k3 = lorenz63_rhs(state + 0.5*dt*k2)
    k4 = lorenz63_rhs(state + dt*k3)
    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def integrate_lorenz(x0, t_end=40.0, dt=0.01):
    steps = int(t_end/dt)
    traj = np.zeros((steps, 3))
    state = x0.copy()
    for i in range(steps):
        traj[i] = state
        state = rk4_step(state, dt)
    return traj

# -------------------------------------------------------
# Nastavitve grafa
# -------------------------------------------------------
os.makedirs("figs", exist_ok=True)

# Začetni pogoj (lahko spremeniš)
x0 = np.array([1.0, 1.0, 1.0])

# Integracija
traj = integrate_lorenz(x0, t_end=40.0, dt=0.01)
X, Y, Z = traj[:,0], traj[:,1], traj[:,2]

t = np.arange(len(X))*0.01

# -------------------------------------------------------
# 1) 3D Lorenz atraktor
# -------------------------------------------------------
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection="3d")
ax.plot(X, Y, Z, lw=0.4, color="black")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Lorenzov atraktor")
plt.tight_layout()
plt.savefig("figs/lorenz_attractor_3D.pdf")
plt.close()

# -------------------------------------------------------
# 2) Časovni poteki
# -------------------------------------------------------
fig, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)

ax[0].plot(t, X); ax[0].set_ylabel("X"); ax[0].grid()
ax[1].plot(t, Y); ax[1].set_ylabel("Y"); ax[1].grid()
ax[2].plot(t, Z); ax[2].set_ylabel("Z"); ax[2].set_xlabel("t"); ax[2].grid()

plt.tight_layout()
plt.savefig("figs/lorenz_timeseries.pdf")
plt.close()

# -------------------------------------------------------
# 3) Projekcija X–Z (kot v poročilih)
# -------------------------------------------------------
plt.figure(figsize=(6,5))
plt.plot(X, Z, lw=0.3, color="darkblue")
plt.xlabel("X")
plt.ylabel("Z")
plt.title("Projekcija Lorenzovega atraktorja: X–Z ravnina")
plt.grid()
plt.tight_layout()
plt.savefig("figs/lorenz_proj_XZ.pdf")
plt.close()

print("Shranjeni grafi v figs/:")
print(" - lorenz_attractor_3D.pdf")
print(" - lorenz_timeseries.pdf")
print(" - lorenz_proj_XZ.pdf")
