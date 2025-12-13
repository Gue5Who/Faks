import numpy as np
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 1. Lorenz integrator (replace with the official one)
# ---------------------------------------------------------
def lorenz63_rhs(t, state, sigma=10.0, r=28.0, b=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (r - z) - y
    dz = x * y - b * z
    return np.array([dx, dy, dz])


def integratorLorenz63(x0, t_end=100.0, dt=0.01):
    """
    Simple RK4 integrator for Lorenz '63.
    If the assignment provides an integrator, replace this.
    """
    steps = int(t_end / dt)
    state = np.array(x0, dtype=float)
    traj = np.zeros((steps, 3))
    t = 0.0

    for i in range(steps):
        k1 = lorenz63_rhs(t, state)
        k2 = lorenz63_rhs(t + dt/2, state + dt*k1/2)
        k3 = lorenz63_rhs(t + dt/2, state + dt*k2/2)
        k4 = lorenz63_rhs(t + dt, state + dt*k3)

        state = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        traj[i] = state
        t += dt

    return traj


# ---------------------------------------------------------
# 2. Generate dataset from many random initial conditions
# ---------------------------------------------------------
def generate_dataset(
        n_trajectories=50,
        t_end=100.0,
        dt=0.01,
        ic_range=(-20, 20)
    ):
    all_data = []

    for i in range(n_trajectories):
        x0 = np.random.uniform(ic_range[0], ic_range[1], size=3)
        traj = integratorLorenz63(x0, t_end=t_end, dt=dt)
        all_data.append(traj)

    data = np.concatenate(all_data, axis=0)
    return data


# ---------------------------------------------------------
# 3. Normalize each variable separately
# ---------------------------------------------------------
def normalize_data(data):
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(data)
    return data_norm, scaler


# ---------------------------------------------------------
# 4. Train/validation split
# ---------------------------------------------------------
def split_train_val(data, train_frac=0.8):
    N = len(data)
    N_train = int(train_frac * N)
    return data[:N_train], data[N_train:]


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Generating dataset...")

    dataset = generate_dataset(
        n_trajectories=50,   # total size ~50*10000 = 500k points
        t_end=100.0,
        dt=0.01
    )

    print("Normalizing data...")
    dataset_norm, scaler = normalize_data(dataset)

    print("Splitting into train/validation...")
    train_data, val_data = split_train_val(dataset_norm)

    print("Saving dataset to file...")
    np.save("data_lorentz/lorenz_train.npy", train_data)
    np.save("data_lorentz/lorenz_val.npy", val_data)
    np.save("data_lorentz/lorenz_scaler_mean.npy", scaler.mean_)
    np.save("data_lorentz/lorenz_scaler_scale.npy", scaler.scale_)

    print("Done!")
