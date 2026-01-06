import matplotlib.pyplot as plt
from greeks.delta_gamma import delta_gamma_mc
from greeks.vega import vega_mc

S0 = 2881
K = 2881
sigma = 0.06023
T = 1
M = 252
N = 100000
r = 0.0232

# Delta & Gamma
x_d, delta, x_g, gamma = delta_gamma_mc(S0, K, sigma, T, M, N, r)

plt.figure(figsize=(10, 6))
plt.plot(x_d, delta)
plt.xlabel("Underlying Price")
plt.ylabel("Delta")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x_g, gamma)
plt.xlabel("Underlying Price")
plt.ylabel("Gamma")
plt.show()

# Vega
x_v, vega = vega_mc(S0, K, sigma, T, M, N, r)

plt.figure(figsize=(10, 6))
plt.plot(x_v, vega)
plt.xlabel("Underlying Price")
plt.ylabel("Vega")
plt.show()
