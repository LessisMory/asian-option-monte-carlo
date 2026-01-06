import numpy as np
from utils.time_grid import generate_observation_indices

def asian_option_mc_correlated(
    S0: float,
    S1: float,
    K: float,
    sigma0: float,
    sigma1: float,
    T: float,
    M: int,
    N: int,
    r: float,
    rho: float
):
    """
    Monte Carlo pricing of correlated two-asset Asian option.

    The payoff asset is a linear combination:
        A_t = 0.5 * S0_t + 1.5 * S1_t
    """
    dt = T / M
    indices = generate_observation_indices(15, M)

    S0_path = np.zeros((M + 1, N))
    S1_path = np.zeros((M + 1, N))
    A = np.zeros((M + 1, N))

    S0_path[0] = S0
    S1_path[0] = S1

    for t in range(1, M + 1):
        z1 = np.random.standard_normal(N)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.standard_normal(N)

        S0_path[t] = S0_path[t - 1] * np.exp(
            (-0.5 * sigma0 ** 2) * dt + sigma0 * np.sqrt(dt) * z1
        )
        S1_path[t] = S1_path[t - 1] * np.exp(
            (-0.5 * sigma1 ** 2) * dt + sigma1 * np.sqrt(dt) * z2
        )

        A[t] = 0.5 * S0_path[t] + 1.5 * S1_path[t]

    payoff = np.maximum(np.mean(A[indices], axis=0) - K, 0)
    price = np.exp(-r * T) * payoff.mean()

    return price
