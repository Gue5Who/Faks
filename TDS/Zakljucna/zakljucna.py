import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from tqdm import tqdm

class BreathingAnnularBilliard:
    def __init__(self, R=2.0, r0=1.0, eps=0.1, omega=0.01):
        """
        R: radij zunanje fiksne stene
        r0: srednji radij notranje stene
        eps: amplituda dihanja notranje stene
        omega: krožna frekvenca dihanja
        """
        self.R = R
        self.r0 = r0
        self.eps = eps
        self.w = omega
        self.T = 2 * np.pi / omega if omega > 0 else 1.0
        
        # Stanje delca
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.t = 0.0

    def set_initial_state(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.t = 0.0

    def angular_momentum(self):
        return self.x * self.vy - self.y * self.vx

    def energy(self):
        return 0.5 * (self.vx**2 + self.vy**2)
    
    def get_inner_radius(self, t):
        return self.r0 * (1 + self.eps * np.cos(self.w * t)) if self.w > 0 else self.r0

    def get_collision_times(self):
        """
        Izračuna čas do trka z zunanjo in notranjo steno.
        """
        # 1. Čas do trka z zunanjo fiksno steno (kvadratna enačba)
        a = self.vx**2 + self.vy**2
        b = 2 * (self.x * self.vx + self.y * self.vy)
        c = self.x**2 + self.y**2 - self.R**2
        
        D = b**2 - 4*a*c
        # Ker delec vedno starta znotraj zunanjega kroga, bo D > 0.
        # Vzamemo pozitivni koren (čas v prihodnosti)
        t_out = (-b + np.sqrt(D)) / (2*a) if D > 0 else float('inf')
        
        # 2. Čas do trka z notranjo steno (numerično z zaščito proti Zenonovemu pojavu)
        # Uporabimo gosto mrežo, da zanesljivo najdemo prvo prečkanje stene
        taus = np.linspace(1e-8, t_out, 3000)
        
        # Razdalja delca od izhodišča
        R_part = np.sqrt((self.x + self.vx*taus)**2 + (self.y + self.vy*taus)**2)
        # Radij dihajoče stene
        R_wall = self.r0 * (1 + self.eps * np.cos(self.w * (self.t + taus)))
        
        F = R_part - R_wall
        
        # Iščemo le eksplicitne prehode OD ZUNAJ NOTER (prepreči zmrzovanje kode na steni)
        crossings = np.where((F[:-1] > 0) & (F[1:] < 0))[0]
        
        t_in = None
        if len(crossings) > 0:
            idx = crossings[0]
            # Najdemo natančno ničlo z Brentovo metodo
            def root_func(tau):
                rp = np.sqrt((self.x + self.vx*tau)**2 + (self.y + self.vy*tau)**2)
                rw = self.r0 * (1 + self.eps * np.cos(self.w * (self.t + tau)))
                return rp - rw
            
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
            # Odboj od fiksne zunanje stene
            self.vx -= 2 * vn * nx
            self.vy -= 2 * vn * ny
        else:
            # Odboj od premikajoče se notranje stene
            # Hitrost stene je odvod radija po času
            V_wall = -self.r0 * self.eps * self.w * np.sin(self.w * self.t)
            
            # Varnostni preklop: Trk velja le, če se delec steni resnično približuje
            if (vn - V_wall) < 0:
                # V sistemu stene se hitrost ohrani po velikosti in obrne smer
                # v_n' - V_wall = -(v_n - V_wall) -> v_n' = 2*V_wall - v_n
                delta_vn = 2 * (V_wall - vn)
                self.vx += delta_vn * nx
                self.vy += delta_vn * ny

    def step_to_time(self, t_target):
        """
        Premika simulacijo točno do časa t_target, vmes rešuje vse trke.
        """
        while self.t < t_target:
            t_out, t_in = self.get_collision_times()
            
            t_event = t_out
            is_inner = False
            if t_in is not None and t_in < t_out:
                t_event = t_in
                is_inner = True
                
            if self.t + t_event > t_target:
                # Naslednji trk je PO t_target. Premaknemo delec in ustavimo loop.
                dt = t_target - self.t
                self.x += self.vx * dt
                self.y += self.vy * dt
                self.t = t_target
            else:
                # Trk se zgodi PRED t_target. Premaknemo delec in se odbijemo.
                self.x += self.vx * t_event
                self.y += self.vy * t_event
                self.t += t_event
                self.perform_collision(is_inner)


def experiment_energy_evolution():
    print("Izvajam eksperiment: Evolucija energije (Adiabatnost vs. Kaos)...")
    
    # PARAMETRI
    R = 2.0
    r0 = 1.0
    eps = 0.2
     
    periods = 50 #prioda nekje 50 je zadostna za kratkotrajno opazovanje in nad 1000 za bezanje navzgor
    
    # 1. Adiabatni primer (Zelo počasno dihanje)
    omega_slow = 0.01
    sim_slow = BreathingAnnularBilliard(R, r0, eps, omega_slow)
    # Fiksiramo začetno stanje. L mora biti konstanten.
    # Začnemo pri zunanjem radiju, y=0. v_y določa vrtilno količino L = R * v_y.
    v0 = 1.0
    vy0 = 0.4  # L = 2.0 * 0.4 = 0.8
    vx0 = -np.sqrt(v0**2 - vy0**2)
    sim_slow.set_initial_state(R, 0, vx0, vy0)
    
    t_vals_slow = []
    E_vals_slow = []
    
    for k in tqdm(range(periods * 50), desc="1/2 Adiabatna simulacija"): # Gledamo veliko period za adiabatno
        target = k * (sim_slow.T / 50) # Zapišemo 50 točk na periodo
        sim_slow.step_to_time(target)
        t_vals_slow.append(sim_slow.t / sim_slow.T)
        E_vals_slow.append(sim_slow.energy())

    # 2. Hiter, ne-adiabaten (kaotičen) primer
    omega_fast = 1.5
    sim_fast = BreathingAnnularBilliard(R, r0, eps, omega_fast)
    sim_fast.set_initial_state(R, 0, vx0, vy0)
    
    t_vals_fast = []
    E_vals_fast = []
    
    for k in tqdm(range(periods * 50), desc="2/2 Kaotična simulacija"):
        target = k * (sim_fast.T / 50)
        sim_fast.step_to_time(target)
        t_vals_fast.append(sim_fast.t / sim_fast.T)
        E_vals_fast.append(sim_fast.energy())

    # Risanje
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(t_vals_slow, E_vals_slow, label=rf'$\omega={omega_slow}$ (Adiabatno)')
    ax1.set_title(r"Adiabatna invarianta: Energija diha skupaj s steno, a se v povprečju ohranja")
    ax1.set_ylabel(r"Kinetična energija $E$")
    ax1.legend()
    ax1.grid()
    
    ax2.plot(t_vals_fast, E_vals_fast, color='red', label=rf'$\omega={omega_fast}$ (Kaotično)')
    ax2.set_title(r"Porušitev invariante (Fermijevo pospeševanje): Energija kaotično raste")
    ax2.set_xlabel(r"Čas [število period $T$]")
    ax2.set_ylabel(r"Kinetična energija $E$")
    ax2.legend()
    ax2.grid()
    
    os.makedirs('figs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'figs/energy_evolution_periods_{periods}.png', dpi=500)
    plt.show()

def experiment_stroboscopic_map():
    print("Izvajam eksperiment: Stroboskopska preslikava...")
    
    R = 2.0
    r0 = 1.0
    eps = 0.2
    omega = 2.0  # Vrnjeno na tvojo vrednost 2.0
    T = 2 * np.pi / omega
    
    # Fiksiramo L (vrtilno količino) za vse orbite
    L_target = 0.8
    vy0 = L_target / R
    
    # Naredili bomo več različnih začetnih hitrosti (energij) za pester fazni portret
    v_start_list = [0.12, 0.32, 0.72, 1.28, 2.00, 3.12] # tvoje prejšnje začetne energije iz grafa
    
    plt.figure(figsize=(10, 10))
    
    for v0 in v_start_list:
        v_vel = np.sqrt(2 * v0) # pretvorba iz energije E_0 v hitrost v0
        vx0 = -np.sqrt(v_vel**2 - vy0**2) if v_vel > vy0 else 0
        if vx0 == 0: continue
            
        sim = BreathingAnnularBilliard(R, r0, eps, omega)
        sim.set_initial_state(R, 0, vx0, vy0)
        
        r_list = []
        vr_list = []
        
        for k in tqdm(range(5000), desc=rf"Orbita E0={v0:.2f}"): # 5000 stroboskopskih iteracij
            sim.step_to_time(k * T)
            
            # Stanje v stroboskopskem trenutku
            r = np.sqrt(sim.x**2 + sim.y**2)
            # Radialna komponenta hitrosti: v_r = (x*v_x + y*v_y) / r
            vr = (sim.x * sim.vx + sim.y * sim.vy) / r
            
            r_list.append(r)
            vr_list.append(vr)
            
        plt.scatter(r_list, vr_list, s=1.5, label=rf'$E_0={v0:.2f}$')
        
    plt.title(fr"Stroboskopska preslikava ob časih $t=n T$" + "\n" + fr"$(\omega={omega}, L={L_target}, R={R}, r_0={r0}, \epsilon={eps})$", fontsize=14)
    plt.xlabel(r"Radij $r$")
    plt.ylabel(r"Radialna hitrost $v_r$")
    # Narišemo meje biljarda (notranja stena se ob stroboskopskih časih nahaja na r0*(1+eps))
    plt.axvline(x=r0*(1+eps), color='black', linestyle='--', label=r"Notranja stena ob $t=nT$")
    plt.axvline(x=R, color='black', linestyle='-', label=r"Zunanja stena")
    plt.legend(loc='upper right', fontsize=10)
    plt.grid()
    os.makedirs('figs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'figs/stroboscopic_map_omega_{omega}.png', dpi=500)
    plt.show()

def multi_omega_stroboscopic():
    print("Izvajam eksperiment: Mreža stroboskopskih preslikav za različne omega...")
    
    R = 2.0
    r0 = 1.0
    eps = 0.2
    L_target = 0.8
    vy0 = L_target / R
    
    v_start_list = [0.12, 0.32, 0.72, 1.28, 2.00, 3.12]
    omegas = [0.1, 0.8, 2.0, 5.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    for i, omega in enumerate(omegas):
        ax = axes[i]
        T = 2 * np.pi / omega
        
        for v0 in v_start_list:
            v_vel = np.sqrt(2 * v0)
            vx0 = -np.sqrt(v_vel**2 - vy0**2) if v_vel > vy0 else 0
            if vx0 == 0: continue
                
            sim = BreathingAnnularBilliard(R, r0, eps, omega)
            sim.set_initial_state(R, 0, vx0, vy0)
            
            r_list = []
            vr_list = []
            
            iterations = 2000 # Malenkost manj iteracij za mrežo, da ne traja predolgo
            for k in tqdm(range(iterations), desc=rf"Omega {omega}, Orbita E0={v0:.2f}", leave=False):
                sim.step_to_time(k * T)
                r = np.sqrt(sim.x**2 + sim.y**2)
                vr = (sim.x * sim.vx + sim.y * sim.vy) / r
                
                r_list.append(r)
                vr_list.append(vr)
                
            ax.scatter(r_list, vr_list, s=1.0, label=rf'$E_0={v0:.2f}$')
            
        ax.set_title(rf"$\omega={omega}$", fontsize=14)
        ax.set_xlabel(r"Radij $r$")
        ax.set_ylabel(r"Radialna hitrost $v_r$")
        ax.axvline(x=r0*(1+eps), color='black', linestyle='--', label=r"Notranja stena")
        ax.axvline(x=R, color='black', linestyle='-', label=r"Zunanja stena")
        ax.grid()
        
    axes[0].legend(loc='upper right', fontsize=10)
    
    plt.suptitle(rf"Porušitev adiabatne invariante in nastanek KAM barier ($L={L_target}, R={R}, r_0={r0}, \epsilon={eps}$)", fontsize=18)
    os.makedirs('figs', exist_ok=True)
    plt.savefig("figs/billiard_bifurcation.png", dpi=500)
    plt.show()

if __name__ == '__main__':
    # experiment_energy_evolution()
    # experiment_stroboscopic_map()
    multi_omega_stroboscopic()