import numpy as np
from utils.time_grid import generate_observation_indices

def asian_option_mc(
    S0: float,
    K: float,
    sigma: float,
    T: float,
    M: int,
    N: int,
    r: float
):
    """
    Monte Carlo pricing of arithmetic Asian option (no correlation).

    Parameters
    ----------
    S0 : float
        Initial asset price
    K : float
        Strike price
    sigma : float
        Volatility
    T : float
        Time to maturity (years)
    M : int
        Number of time steps
    N : int
        Number of Monte Carlo paths
    r : float
        Risk-free rate

    Returns
    -------
    float
        Option price
    """
    dt = T / M
    indices = generate_observation_indices(15, M)

    S = np.zeros((M + 1, N))
    S[0] = S0

    for t in range(1, M + 1):
        z = np.random.standard_normal(N)
        S[t] = S[t - 1] * np.exp(
            (-0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        )

    payoff = np.maximum(np.mean(S[indices], axis=0) - K, 0)
    price = np.exp(-r * T) * payoff.mean()

    return price
