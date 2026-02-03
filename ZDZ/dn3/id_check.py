from pygadgetreader import readsnap

snap = "kroglasta/output2/snapshot_000"

pos = readsnap(snap, "pos", 3)   # bulge = type 3
pid = readsnap(snap, "pid", 3)   # particle IDs (type 3)

print("Positions shape:", pos.shape)
print("First 20 IDs:", pid[:20])
print("Min ID:", pid.min(), "Max ID:", pid.max())
