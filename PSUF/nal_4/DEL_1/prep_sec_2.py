import numpy as np

def create_sequences(data, input_seq_len):
    """
    Converts a time series of shape (N, 3) into sequences for many-to-one training.

    Parameters:
        data : array of shape (N, 3)
            The normalized Lorenz-63 time series.
        input_seq_len : int
            Length of the input sequence (k in assignment).

    Returns:
        X : array of shape (num_samples, input_seq_len, 3)
        y : array of shape (num_samples, 3)
    """
    X, y = [], []
    N = len(data)

    for i in range(N - input_seq_len):
        seq_x = data[i : i + input_seq_len]       # sequence of k points
        seq_y = data[i + input_seq_len]           # the next point: X(t+Δt)

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)
