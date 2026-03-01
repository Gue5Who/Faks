#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# Helpers: kinematics
# -------------------------
def advance_const_accel(x, v, a, dt):
    """Advance (x,v) under constant accel a for time dt."""
    x_new = x + v*dt + 0.5*a*dt*dt
    v_new = v + a*dt
    return x_new, v_new

def time_to_wall(x, v, a, wall):
    """
    Smallest positive time t to reach x(t)=wall under constant accel:
        x + v t + 1/2 a t^2 = wall
    Returns np.inf if no positive solution.
    """
    # Solve 0.5 a t^2 + v t + (x - wall) = 0
    A = 0.5*a
    B = v
    C = x - wall

    if abs(A) < 1e-14:  # near zero acceleration -> linear
        if abs(B) < 1e-14:
            return np.inf
        t = -C / B
        return t if t > 1e-14 else np.inf

    disc = B*B - 4*A*C
    if disc < 0:
        return np.inf
    sqrt_disc = np.sqrt(disc)

    t1 = (-B - sqrt_disc) / (2*A)
    t2 = (-B + sqrt_disc) / (2*A)

    t_candidates = [t for t in (t1, t2) if t > 1e-14]
    return min(t_candidates) if t_candidates else np.inf


# -------------------------
# Event-driven simulator
# -------------------------
def simulate_strobe(
    k,
    x0=0.73,
    p0s=None,
    n_periods=5000,
    record_each_period=True
):
    """
    Dimensionless system:
      x in [0,1]
      p = v (since we scaled m out): p_hat = tau*p/(a m) = tau*v/a
      time s = t/tau  so one period = 1

    Field:
      E = +E0 on (n, n+1/2), E = -E0 on (n+1/2, n+1)
    Physical eq:
      dp/ds = -sigma*k , sigma=+1 first half, sigma=-1 second half
      dx/ds = p

    We record (x,p) at stroboscopic times s = n (each full period).
    """
    if p0s is None:
        p0s = np.linspace(0, 10, 50)

    # Storage per trajectory for coloring
    xs = [[] for _ in range(len(p0s))]
    ps = [[] for _ in range(len(p0s))]

    for i, p0 in enumerate(p0s):
        x = float(x0)
        p = float(p0)
        s = 0.0  # current time in units of tau

        # record initial strobe at s=0
        xs[i].append(x)
        ps[i].append(p)

        for n in range(n_periods):
            # We want to advance from s=n to s=n+1 with exact event handling
            target = (n + 1) * 1.0

            while s < target - 1e-15:
                # Determine current field sign sigma based on phase within period
                phase = s - np.floor(s)  # in [0,1)
                if phase < 0.5:
                    sigma = +1
                    next_switch = np.floor(s) + 0.5
                else:
                    sigma = -1
                    next_switch = np.floor(s) + 1.0

                a = -sigma * k  # dp/ds

                # Time until switch and until end-of-period target
                dt_switch = next_switch - s
                dt_target = target - s
                dt_cap = min(dt_switch, dt_target)

                # Time to hit left/right wall within this constant-accel segment
                t_left = time_to_wall(x, p, a, wall=0.0)
                t_right = time_to_wall(x, p, a, wall=1.0)
                t_wall = min(t_left, t_right)

                # Next event is min(wall hit, dt_cap)
                dt = min(t_wall, dt_cap)

                # Advance
                x, p = advance_const_accel(x, p, a, dt)
                s += dt

                # If wall hit happened first, reflect elastically: p -> -p
                if t_wall < dt_cap - 1e-12:
                    # Clamp x numerically to wall
                    if x < 0.5:
                        x = 0.0
                    else:
                        x = 1.0
                    p = -p

            # Record stroboscopic point at end of the period
            if record_each_period:
                xs[i].append(x)
                ps[i].append(p)

    xs = [np.array(v) for v in xs]
    ps = [np.array(v) for v in ps]
    return xs, ps, p0s


# -------------------------
# Plotting (colored by p0)
# -------------------------
def plot_colored(xs, ps, p0s, k, x0, out_prefix="portrait", save=True):
#    Path("figs").mkdir(exist_ok=True)
#
#    fig = plt.figure(figsize=(8, 10))
#    cmap = plt.cm.viridis
#    norm = plt.Normalize(p0s.min(), p0s.max())
#
#    for x, p, p0 in zip(xs, ps, p0s):
#        plt.scatter(
#            x, p,
#            s=0.35,
#            alpha=0.6,
#            color=cmap(norm(p0)),
#            rasterized=True,
#            linewidths=0
#        )
#
#    plt.xlim(0, 1)
#    plt.xlabel(r"$\hat{x}$")
#    plt.ylabel(r"$\hat{p}$")
#    plt.title(rf"Stroboscopic phase portrait, $k={k:.3g}$  ($x_0={x0:.4f}$)")
#
#    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#    cb = plt.colorbar(sm)
#    cb.set_label(r"initial $p_0$")
#
#    plt.tight_layout()
#
#    if save:
#        pdf = Path("figs") / f"{out_prefix}_k_{k:.3g}.pdf"
#        fig.savefig(pdf)
#        print("Saved:", pdf)
#
#    #plt.show()
#    plt.close(fig)

    #diskretne barve
    Path("figs").mkdir(exist_ok=True)

    fig = plt.figure(figsize=(8, 10))

    cmap = plt.cm.get_cmap("tab20", len(p0s))  # 20 distinct colors (cycled if >20)

    for i, (x, p) in enumerate(zip(xs, ps)):
        plt.scatter(
            x, p,
            s=0.35,
            alpha=0.7,
            color=cmap(i),
            rasterized=True,
            linewidths=0
        )

    plt.xlim(0, 1)
    plt.xlabel(r"$\hat{x}$")
    plt.ylabel(r"$\hat{p}$")
    plt.title(rf"Phase portrait, $k={k:.3g}$  ($x_0={x0:.4f}$)")

    plt.tight_layout()

    if save:
        pdf = Path("figs") / f"{out_prefix}_k_{k:.3g}.pdf"
        fig.savefig(pdf)
        print("Saved:", pdf)

    #plt.show()
    plt.close(fig)

def main():
    x0 = 0.6122448979591836
    p0s = np.linspace(0, 10, 100)

    # Increase n_periods for denser/clearer islands (like colleague).
    # Start with 5000; go to 50000 if you want very dense plots.
    n_periods = 15000

    k_values = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    for k in k_values:
        xs, ps, p0s_used = simulate_strobe(
            k=k,
            x0=x0,
            p0s=p0s,
            n_periods=n_periods,
            record_each_period=True
        )
        plot_colored(xs, ps, p0s_used, k, x0, out_prefix="portrait_event", save=True)


if __name__ == "__main__":
    main()