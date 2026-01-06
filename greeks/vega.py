import numpy as np
from utils.time_grid import generate_observation_indices

def vega_mc(
    S0, K, sigma, T, M, N, r, bump=0.01
):
    """
    Vega estimation via volatility bump-and-revalue.
    """
    dt = T / M
    indices = generate_observation_indices(15, M)

    vegas = []
    grid = [S0 - 2000 + i * 20 for i in range(200)]

    for s in grid:
        np.random.seed(2000)
        S = np.zeros((M + 1, N))
        S_bump = np.zeros((M + 1, N))
        S[0] = S_bump[0] = s

        for t in range(1, M + 1):
            z = np.random.standard_normal(N)
            S[t] = S[t - 1] * np.exp(
                (-0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
            )
            S_bump[t] = S_bump[t - 1] * np.exp(
                (-0.5 * (sigma + bump) ** 2) * dt + (sigma + bump) * np.sqrt(dt) * z
            )

        price = np.exp(-r * T) * np.maximum(np.mean(S[indices], axis=0) - K, 0).mean()
        price_bump = np.exp(-r * T) * np.maximum(np.mean(S_bump[indices], axis=0) - K, 0).mean()

        vegas.append((price_bump - price) / bump)

    return grid, vegas
