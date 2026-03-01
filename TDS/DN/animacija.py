import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

# ----------------------------
# Physics in dimensionless units
# ----------------------------
# x in [0, a_hat] where a_hat = 1
# time s = t/tau
# p_hat = dx/ds
# dp/ds = -sigma * k, sigma = +1 for E>0, -1 for E<0
# E switches every half period: s = n/2

def time_to_wall(x, v, a, wall):
    """
    Smallest positive t solving x + v t + 0.5 a t^2 = wall.
    Returns np.inf if no positive solution.
    """
    A = 0.5 * a
    B = v
    C = x - wall

    if abs(A) < 1e-14:
        if abs(B) < 1e-14:
            return np.inf
        t = -C / B
        return t if t > 1e-14 else np.inf

    disc = B*B - 4*A*C
    if disc < 0:
        return np.inf
    sdisc = np.sqrt(disc)
    t1 = (-B - sdisc) / (2*A)
    t2 = (-B + sdisc) / (2*A)
    ts = [t for t in (t1, t2) if t > 1e-14]
    return min(ts) if ts else np.inf

def advance_const_accel(x, v, a, dt):
    """Exact kinematics under constant acceleration a for time dt."""
    x_new = x + v*dt + 0.5*a*dt*dt
    v_new = v + a*dt
    return x_new, v_new

def sigma_E(s):
    """+1 for first half of period, -1 for second half; period = 1."""
    phase = s - np.floor(s)
    return +1 if phase < 0.5 else -1

def next_switch_time(s):
    """Next time (>=s) when E flips sign (every 0.5)."""
    n = np.floor(2*s)  # integer half-period index
    return (n + 1) / 2.0

def advance_with_events(x, v, s, dt, k, xmin=0.0, xmax=1.0):
    """
    Advance state by dt while handling:
      - E-field switches
      - wall collisions
    Event-driven (exact).
    """
    remaining = dt
    while remaining > 1e-12:
        sig = sigma_E(s)
        a = -sig * k

        t_sw = next_switch_time(s) - s
        t_hit_left = time_to_wall(x, v, a, xmin)
        t_hit_right = time_to_wall(x, v, a, xmax)
        t_wall = min(t_hit_left, t_hit_right)

        # Next event within remaining time?
        t_event = min(t_sw, t_wall, remaining)

        # advance to event (or end)
        x, v = advance_const_accel(x, v, a, t_event)
        s += t_event
        remaining -= t_event

        # If wall hit occurred first (strictly before switch/end), reflect
        if t_wall < min(t_sw, t_event + 1e-12) and t_wall <= t_event + 1e-12:
            # Clamp due to numeric error
            if abs(x - xmin) < abs(x - xmax):
                x = xmin
            else:
                x = xmax
            v = -v

        # If switch happened, sigma_E(s) will change automatically next loop

    return x, v, s


# ----------------------------
# Animation setup
# ----------------------------
def animate_particle(k=1.0, x0=0.73, p0=3.0, tau=1.0,
                     fps=60, seconds=10, y_particle=0.0,
                     save=False, outname="figs/particle_flight.mp4"):
    """
    k: dimensionless strength (k = e E0 tau^2 / (a m))
    x0 in [0,1], p0 is dimensionless momentum (dx/ds)
    fps, seconds control animation duration
    save: if True save mp4 (requires ffmpeg) or gif depending on extension
    """

    Path("figs").mkdir(exist_ok=True)

    # time step per frame in dimensionless s-units
    dt = 1.0 / fps  # because period=1 in s-units; feels nice visually
    nframes = int(seconds * fps)

    # initial state
    x = float(x0)
    v = float(p0)
    s = 0.0

    # Figure
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.35, 0.55)
    ax.set_yticks([])
    ax.set_xlabel(r"$\hat{x} = x/a$")
    ax.set_title(rf"Particle in a 1D well (k={k}, x0={x0}, p0={p0})")

    # Walls
    ax.plot([0, 0], [-0.25, 0.25], lw=4)
    ax.plot([1, 1], [-0.25, 0.25], lw=4)

    # Floor line
    ax.plot([0, 1], [y_particle, y_particle], lw=1, alpha=0.5)

    # Particle marker
    (dot,) = ax.plot([x], [y_particle], marker="o", markersize=10)

    # E-field arrow (we’ll redraw via annotate-like arrow patch)
    # Use a single FancyArrowPatch-like via annotate for simplicity
    arrow_y = 0.25
    arrow = ax.annotate(
        "", xy=(0.75, arrow_y), xytext=(0.25, arrow_y),
        arrowprops=dict(arrowstyle="->", lw=2)
    )

    # Text readout
    text = ax.text(0.02, 0.85, "", transform=ax.transAxes)

    def update(frame):
        nonlocal x, v, s

        # advance one frame with exact events
        x, v, s = advance_with_events(x, v, s, dt, k, xmin=0.0, xmax=1.0)

        # update particle
        dot.set_data([x], [y_particle])

        # update arrow direction to match E
        sig = sigma_E(s)
        if sig > 0:
            # E points right
            arrow.xy = (0.78, arrow_y)
            arrow.xyann = (0.22, arrow_y)
            arrow.set_text("")  # keep empty
            arrow.arrowprops.update(dict(arrowstyle="->", lw=2))
        else:
            # E points left
            arrow.xy = (0.22, arrow_y)
            arrow.xyann = (0.78, arrow_y)
            arrow.set_text("")
            arrow.arrowprops.update(dict(arrowstyle="->", lw=2))

        text.set_text(
            rf"$s=t/\tau={s:.2f}$   "
            rf"$\hat x={x:.3f}$   "
            rf"$\hat p={v:.3f}$   "
            + ("E →" if sig > 0 else "E ←")
        )

        return dot, arrow, text

    ani = FuncAnimation(fig, update, frames=nframes, interval=1000/fps, blit=False)

    if save:
        outpath = Path(outname)
        outpath.parent.mkdir(exist_ok=True)

        # mp4 needs ffmpeg; gif uses pillow
        if outpath.suffix.lower() == ".gif":
            ani.save(outpath, writer="pillow", fps=fps)
        else:
            ani.save(outpath, writer="ffmpeg", fps=fps)
        print("Saved:", outpath)

    return ani


# --- Run interactively ---
# In Jupyter: just call animate_particle(...) and the animation will display (usually).
ani = animate_particle(k=1.0, x0=0.73, p0=1.0, fps=60, seconds=30, save=True)
plt.show()