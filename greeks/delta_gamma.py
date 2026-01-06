import numpy as np
from utils.time_grid import generate_observation_indices

def delta_gamma_mc(
    S0, K, sigma, T, M, N, r
):
    """
    Numerical Delta and Gamma estimation
    using finite differences and common random numbers.
    """
    dt = T / M
    indices = generate_observation_indices(15, M)

    prices = []
    grid = [S0 - 2000 + i * 20 for i in range(200)]

    for s in grid:
        np.random.seed(2000)
        S = np.zeros((M + 1, N))
        S[0] = s

        for t in range(1, M + 1):
            z = np.random.standard_normal(N)
            S[t] = S[t - 1] * np.exp(
                (-0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
            )

        payoff = np.maximum(np.mean(S[indices], axis=0) - K, 0)
        price = np.exp(-r * T) * payoff.mean()
        prices.append(price)

    delta = [(prices[i + 2] - prices[i]) / 40 for i in range(len(prices) - 2)]
    gamma = [(delta[i + 2] - delta[i]) / 40 for i in range(len(delta) - 2)]

    return grid[1:-1], delta, grid[3:-3], gamma
