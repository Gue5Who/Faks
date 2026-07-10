import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import brentq
import os

# ---------------------------------------------------------
# Isti fizikalni pogon kot prej (BreathingAnnularBilliard)
# ---------------------------------------------------------
class BreathingAnnularBilliard:
    def __init__(self, R=2.0, r0=1.0, eps=0.1, omega=0.01):
        self.R = R
        self.r0 = r0
        self.eps = eps
        self.w = omega
        self.T = 2 * np.pi / omega if omega > 0 else 1.0
        self.x, self.y = 0.0, 0.0
        self.vx, self.vy = 0.0, 0.0
        self.t = 0.0

    def set_initial_state(self, x, y, vx, vy):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.t = 0.0

    def get_inner_radius(self, t):
        return self.r0 * (1 + self.eps * np.cos(self.w * t))

    def get_collision_times(self):
        # 1. Zunanja fiksna stena
        a = self.vx**2 + self.vy**2
        b = 2 * (self.x * self.vx + self.y * self.vy)
        c = self.x**2 + self.y**2 - self.R**2
        D = b**2 - 4*a*c
        t_out = (-b + np.sqrt(D)) / (2*a) if D > 0 else float('inf')
        
        # 2. Notranja dihajoča stena
        taus = np.linspace(1e-8, t_out, 1000)
        rp = np.sqrt((self.x + self.vx*taus)**2 + (self.y + self.vy*taus)**2)
        rw = self.r0 * (1 + self.eps * np.cos(self.w * (self.t + taus)))
        F = rp - rw
        
        crossings = np.where((F[:-1] > 0) & (F[1:] < 0))[0]
        
        t_in = None
        if len(crossings) > 0:
            idx = crossings[0]
            def root_func(tau):
                rp_ = np.sqrt((self.x + self.vx*tau)**2 + (self.y + self.vy*tau)**2)
                rw_ = self.r0 * (1 + self.eps * np.cos(self.w * (self.t + tau)))
                return rp_ - rw_
            try:
                t_in = brentq(root_func, taus[idx], taus[idx+1])
            except ValueError:
                t_in = (taus[idx] + taus[idx+1]) / 2.0
                
        return t_out, t_in

    def perform_collision(self, is_inner):
        r_mag = np.sqrt(self.x**2 + self.y**2)
        nx = self.x / r_mag
        ny = self.y / r_mag
        vn = self.vx * nx + self.vy * ny
        
        if not is_inner:
            self.vx -= 2 * vn * nx
            self.vy -= 2 * vn * ny
        else:
            V_wall = -self.r0 * self.eps * self.w * np.sin(self.w * self.t)
            if (vn - V_wall) < 0:
                delta_vn = 2 * (V_wall - vn)
                self.vx += delta_vn * nx
                self.vy += delta_vn * ny

    def step_to_time(self, t_target):
        while self.t < t_target:
            t_out, t_in = self.get_collision_times()
            
            t_event = t_out
            is_inner = False
            if t_in is not None and t_in < t_out:
                t_event = t_in
                is_inner = True
                
            if self.t + t_event > t_target:
                dt = t_target - self.t
                self.x += self.vx * dt
                self.y += self.vy * dt
                self.t = t_target
            else:
                self.x += self.vx * t_event
                self.y += self.vy * t_event
                self.t += t_event
                self.perform_collision(is_inner)

# ---------------------------------------------------------
# Koda za animacijo
# ---------------------------------------------------------
def animate_billiard(R = 2.0, r0 = 1.0, eps = 0.2, omega = 2.0, E0 = 0.72):
    # Parametri biljarda
    R = R
    r0 = r0
    eps = eps
    omega = omega  # Frekvenca dihanja

    
    # Parametri delca
    L_target = 0.8
    vy0 = L_target / R
    E0 = E0 # Izberi eno od energij iz prejšnjih grafov
    v0_vel = np.sqrt(2 * E0)
    vx0 = -np.sqrt(v0_vel**2 - vy0**2)
    
    sim = BreathingAnnularBilliard(R, r0, eps, omega)
    sim.set_initial_state(R, 0, vx0, vy0)

    # Nastavitve animacije
    fps = 30
    dt = 0.05       # Časovni korak med posameznimi sličicami
    trail_length = 60 # Koliko preteklih točk rišemo (dolžina repa)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-R - 0.2, R + 0.2)
    ax.set_ylim(-R - 0.2, R + 0.2)
    ax.set_aspect('equal')
    ax.set_title(rf"Dihajoči biljard ($\omega={omega}$, $E_0={E0}$)")
    ax.axis('off') # Skrijemo osi, da izgleda bolj "igričarsko"

    # Grafični elementi
    # 1. Zunanja fiksna stena
    theta = np.linspace(0, 2*np.pi, 200)
    outer_wall, = ax.plot(R * np.cos(theta), R * np.sin(theta), color='black', lw=2)
    
    # 2. Notranja dihajoča stena
    inner_wall, = ax.plot([], [], color='blue', lw=2)
    
    # 3. Sled delca
    trail, = ax.plot([], [], color='red', lw=1, alpha=0.6)
    
    # 4. Delec
    particle, = ax.plot([], [], 'ro', markersize=6)

    # Shranjevanje preteklih lokacij za rep
    history_x = []
    history_y = []

    def init():
        inner_wall.set_data([], [])
        trail.set_data([], [])
        particle.set_data([], [])
        return inner_wall, trail, particle

    def update(frame):
        # 1. Premaknemo simulacijo naprej za dt
        target_t = frame * dt
        sim.step_to_time(target_t)
        
        # 2. Posodobimo sled (zgodovino)
        history_x.append(sim.x)
        history_y.append(sim.y)
        if len(history_x) > trail_length:
            history_x.pop(0)
            history_y.pop(0)
        
        # 3. Posodobimo pozicijo delca in sledi
        particle.set_data([sim.x], [sim.y])
        trail.set_data(history_x, history_y)
        
        # 4. Posodobimo notranjo steno
        current_r = sim.get_inner_radius(sim.t)
        inner_wall.set_data(current_r * np.cos(theta), current_r * np.sin(theta))
        
        return inner_wall, trail, particle

    # Ustvarimo animacijo
    frames_total = 3000 # Število sličic (20 sekund pri 30 FPS)
    ani = animation.FuncAnimation(fig, update, frames=frames_total,
                                  init_func=init, blit=True, interval=1000/fps)

    plt.tight_layout()
    
    # Če želiš shraniti kot GIF, odkomentiraj spodnji vrstici (potrebuješ knjižnico Pillow):
    # print("Shranjujem animacijo kot GIF...")
    # ani.save('billiard_animation.gif', writer='pillow', fps=fps)
    
    # Če želiš shraniti kot MP4 (potrebuješ ffmpeg nameščen na sistemu):
    print("Shranjujem animacijo kot MP4...")
    os.makedirs("gifs", exist_ok=True)
    ani.save(f'gifs/billiard_animation_{omega}_{E0}.mp4', writer='ffmpeg', fps=fps)
    # ce delam for loop na na  odpre plota in mi jih naj samo shrani in zapre
    plt.close() 

if __name__ == '__main__':
    print("Zaganjam animacijo...")
    #animate_billiard(omega=0.1, eps=0.1, E0=0.72)

    #for loop ki izrise animacije za razlicne omege in energije
    omegas = [0.1, 0.5, 1.0, 2.0]
    energies = [0.5, 1.0, 1.5, 2.0]
    for omega in omegas:
        for E0 in energies:
            animate_billiard(omega=omega, eps=0.2, E0=E0)
            