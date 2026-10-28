"""Chapter 8: Shapley values by hand for a toy 2-feature model (example numbers)."""
from itertools import permutations

# toy model: A = "qty < 300", B = "category is Passive"; reference point = (0, 0)
f = {(0, 0): 1.0, (1, 0): 1.6, (0, 1): 0.2, (1, 1): 0.4}
x = (1, 1)
phi = [0.0, 0.0]
orders = list(permutations([0, 1]))
for order in orders:
    z = [0, 0]
    for j in order:
        before = f[tuple(z)]
        z[j] = x[j]
        phi[j] += (f[tuple(z)] - before) / len(orders)
print(f"base f(0,0) = {f[(0, 0)]:.2f}")
print(f"phi_A = {phi[0]:+.2f}  phi_B = {phi[1]:+.2f}")
print(f"base + phi_A + phi_B = {f[(0, 0)] + sum(phi):.2f}  model f(1,1) = {f[x]:.2f}")
