import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from pygadgetreader import readsnap


def find_snapshots(folder: str, base: str = "snapshot_") -> list[str]:
    files = glob.glob(os.path.join(folder, f"{base}*"))
    files = [f for f in files if os.path.basename(f).startswith(base)]

    def key_fn(path):
        import re
        m = re.search(r"(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else 0

    return sorted(files, key=key_fn)


def load_pos(snapfile: str, ptype) -> np.ndarray:
    """ptype can be int (recommended)"""
    try:
        pos = readsnap(snapfile, "pos", ptype)
        if pos is None:
            return np.zeros((0, 3))
        return pos
    except Exception as e:
        print(f"[warn] {snapfile}: couldn't read ptype={ptype} pos ({e})")
        return np.zeros((0, 3))


def load_pid(snapfile: str, ptype) -> np.ndarray:
    """Try to read particle IDs for ptype. Fallback pid -> id."""
    try:
        pid = readsnap(snapfile, "pid", ptype)
        if pid is None:
            raise RuntimeError("pid returned None")
        return pid
    except Exception:
        try:
            pid = readsnap(snapfile, "id", ptype)
            if pid is None:
                raise RuntimeError("id returned None")
            return pid
        except Exception as e:
            print(f"[warn] {snapfile}: couldn't read IDs (pid/id) for ptype={ptype} ({e})")
            return np.zeros((0,), dtype=np.int64)


def robust_lim(pos: np.ndarray, view: str, q: float = 0.995) -> float:
    if pos.shape[0] == 0:
        return 1.0
    if view == "faceon":
        xy = pos[:, [0, 1]]
    else:
        xy = pos[:, [0, 2]]
    r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    lim = float(np.quantile(r, q))
    return max(lim, 1.0)


def save_animation(anim, outpath: str, fps: int = 20):
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    try:
        anim.save(outpath, writer=FFMpegWriter(fps=fps, bitrate=3000))
        print(f"[ok] Saved MP4: {outpath}")
    except Exception as e:
        gif_path = os.path.splitext(outpath)[0] + ".gif"
        print(f"[warn] MP4 failed ({e}); saving GIF: {gif_path}")
        anim.save(gif_path, writer=PillowWriter(fps=fps))
        print(f"[ok] Saved GIF: {gif_path}")


def make_anim(folder: str, outpath: str, view: str = "faceon",
             fps: int = 20, stride: int = 1,
             s_disk: float = 0.20, s_gas: float = 0.20,
             a_disk: float = 0.12, a_gas: float = 0.12,
             s_gc: float = 1.0, a_gc: float = 0.6,
             phi: float = 0.0,
             theta: float = 0.0,
             GC_ID_MIN: int = 145000):
    """
    Galaxy: disk (ptype 2) + gas (ptype 0)
    GC: bulge (ptype 3) with ID > GC_ID_MIN
    """

    snaps = find_snapshots(folder)[::stride]
    if not snaps:
        raise RuntimeError(f"No snapshots found in {folder}")

    # Use ints for ptypes (pygadgetreader on your setup expects ints)
    PT_GAS = 0
    PT_DISK = 2
    PT_BULGE = 3  # GC lives here in your merged run

    # Load first frame to set limits (disk+gas combined)
    pos_disk0 = load_pos(snaps[0], PT_DISK)
    pos_gas0  = load_pos(snaps[0], PT_GAS)
    pos0 = np.vstack([p for p in (pos_disk0, pos_gas0) if p.shape[0] > 0])

    lim = robust_lim(pos0, view=view, q=0.995)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", "box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]" if view == "faceon" else "z [kpc]")

    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    # Three scatters: disk + gas + GC
    sc_disk = ax.scatter([], [], s=s_disk, alpha=a_disk, rasterized=True, label="disk")
    sc_gas  = ax.scatter([], [], s=s_gas,  alpha=a_gas,  rasterized=True, label="gas")
    sc_gc   = ax.scatter([], [], s=s_gc,   alpha=a_gc,   rasterized=True, label="GC (ptype3, ID>145000)")

    ax.legend(loc="lower right", frameon=True)

    def project(pos: np.ndarray) -> np.ndarray:
        if pos.shape[0] == 0:
            return np.empty((0, 2))
        if view == "faceon":
            return pos[:, [0, 1]]
        return pos[:, [0, 2]]

    def init():
        sc_disk.set_offsets(np.empty((0, 2)))
        sc_gas.set_offsets(np.empty((0, 2)))
        sc_gc.set_offsets(np.empty((0, 2)))
        title.set_text("")
        return sc_disk, sc_gas, sc_gc, title

    def update(i):
        snap = snaps[i]

        pos_disk = load_pos(snap, PT_DISK)
        pos_gas  = load_pos(snap, PT_GAS)

        xy_disk = project_rotated(pos_disk, phi, theta)
        xy_gas  = project_rotated(pos_gas,  phi, theta)
        # GC: type 3 filtered by ID
        pos_bulge = load_pos(snap, PT_BULGE)
        pid_bulge = load_pid(snap, PT_BULGE)

        if pos_bulge.shape[0] > 0 and pid_bulge.shape[0] == pos_bulge.shape[0]:
            gc_mask = (pid_bulge > GC_ID_MIN)
            xy_gc   = project_rotated(pos_bulge[gc_mask], phi, theta)
        else:
            xy_gc = np.empty((0, 2))

        # recenter using disk+gas median (keeps galaxy centered)
        xy_all = np.vstack([a for a in (xy_disk, xy_gas) if a.shape[0] > 0])
        if xy_all.shape[0] > 0:
            c = np.median(xy_all, axis=0)
            xy_disk = xy_disk - c
            xy_gas  = xy_gas - c
            xy_gc   = xy_gc - c

        sc_disk.set_offsets(xy_disk)
        sc_gas.set_offsets(xy_gas)
        sc_gc.set_offsets(xy_gc)

        title.set_text(f"{os.path.basename(snap)}  ({i+1}/{len(snaps)})")
        return sc_disk, sc_gas, sc_gc, title

    anim = FuncAnimation(fig, update, frames=len(snaps), init_func=init, blit=True)
    save_animation(anim, outpath, fps=fps)
    plt.close(fig)

def project_rotated(pos, phi_deg, theta_deg):
    """
    Rotate positions by:
      - phi around z-axis
      - theta around x-axis
    then project to 2D (X', Y')
    """
    if pos.shape[0] == 0:
        return np.empty((0, 2))

    phi = np.deg2rad(phi_deg)
    theta = np.deg2rad(theta_deg)

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

    # rotation around z-axis (phi)
    x1 =  np.cos(phi)*x - np.sin(phi)*y
    y1 =  np.sin(phi)*x + np.cos(phi)*y
    z1 =  z

    # rotation around x-axis (theta)
    x2 = x1
    y2 =  np.cos(theta)*y1 - np.sin(theta)*z1
    z2 =  np.sin(theta)*y1 + np.cos(theta)*z1

    return np.column_stack((x2, y2))



if __name__ == "__main__":
    SNAP_DIR = "./kroglasta/output2"   # <- nastavi na mapo za MW+GC run
    NAME = "MW_GC2"
    OUT_MP4_FACEON = "./Animacije/" + NAME + "_faceon.mp4"
    OUT_MP4_EDGEON = "./Animacije/" + NAME + "_edgeon.mp4"

    make_anim(
        SNAP_DIR, OUT_MP4_FACEON, view="faceon", fps=10, stride=1,
        s_disk=0.3, s_gas=0.18, a_disk=0.80, a_gas=0.70,
        s_gc=1.2, a_gc=0.6,
        GC_ID_MIN=145000,
        phi = 0, theta = 70
    )

    make_anim(
        SNAP_DIR, OUT_MP4_EDGEON, view="edgeon", fps=10, stride=1,
        s_disk=0.3, s_gas=0.18, a_disk=0.80, a_gas=0.70,
        s_gc=1.2, a_gc=0.6,
        GC_ID_MIN=145000,
        phi = 0, theta = 160
    )
