import numpy as np
import matplotlib.pyplot as plt
from pygadgetreader import readsnap

snap = "kroglasta/output2/snapshot_000"

# preberi bulge
pos = readsnap(snap, "pos", 3)
pid = readsnap(snap, "pid", 3)

# maske
mw = (pid >= 130001) & (pid <= 145000)
gc = (pid > 145000)

# projekcija (face-on)
xy_mw = pos[mw][:, [0, 1]]
xy_gc = pos[gc][:, [0, 1]]

# recentriraj na MW bulge
center = np.median(xy_mw, axis=0)
xy_mw -= center
xy_gc -= center

plt.figure(figsize=(6, 6))

plt.scatter(
    xy_mw[:, 0], xy_mw[:, 1],
    s=0.15, alpha=0.05,
    label="MW bulge",
    rasterized=True
)

plt.scatter(
    xy_gc[:, 0], xy_gc[:, 1],
    s=1.0, alpha=1,
    label="Globular Cluster",
    rasterized=True
)

plt.xlabel("x [kpc]")
plt.ylabel("y [kpc]")
plt.title("Initial configuration: MW + GC")
plt.legend(loc="upper right")
plt.gca().set_aspect("equal", "box")
plt.tight_layout()
plt.show()
