import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Constants given in the problem
beta = 0.0075
lambda_ = 0.08
Lambda_ = 6e-5
N0 = 1
C0 = beta * N0 / (Lambda_ * lambda_)
alpha = 2.0 * beta
Nref = 1.0
rho0 = -6.67 * beta + alpha * N0 / Nref

# Reactivity function with negative feedback
def reactivity(t, N):
    return rho0 - alpha * N / Nref

# Define the system of equations
def point_kinetics(N, C, rho):
    dN_dt = (rho - beta) / Lambda_ * N + lambda_ * C
    dC_dt = beta / Lambda_ * N - lambda_ * C
    return np.array([dN_dt, dC_dt])

# The A matrix function as in A(t, f)
def A_matrix(t, f):
    rho = reactivity(t, f[0])
    return np.array([[(rho - beta) / Lambda_, lambda_], [beta / Lambda_, -lambda_]])

# The norm function as defined in the problem
def norm(f):
    return np.sqrt(f[0] ** 2 + f[1] ** 2)

# Lagged Trapezoidal/BDF-2 scheme
def lagged_TBDF2(t, dt):
    # Initial conditions
    f = np.array([N0, C0])

    # Storing the solution
    f_values = [f]

    for i in range(1, len(t)):
        A_n = A_matrix(t[i - 1], f)
        A_half = A_matrix(t[i] - dt / 2, f)

        # Inverse calculation for intermediate step
        inv_matrix = inv(np.identity(2) - A_half * dt / 4)
        f_half = inv_matrix @ ((np.identity(2) + A_n * dt / 4) @ f)
        A_next = A_matrix(t[i], f_half)

        # Inverse calculation for final step
        inv_matrix = inv(3 * np.identity(2) - A_next * dt)
        f_next = inv_matrix @ (4 * f_half - f)
        f = f_next
        f_values.append(f)

    return np.array(f_values)

# Fully-iterated Picard TBDF-2 scheme
def picard_TBDF2(t, dt, tol=1e-8, max_iter=100):
    # Initial conditions
    f = np.array([N0, C0])

    # Storing the solution
    f_values = [f]

    for i in range(1, len(t)):

        f_prev = f_values[-1]
        f_half = f_prev

        for _ in range(max(2, max_iter)):

            f_half_prev = f_half
            A_half = A_matrix(t[i] - dt / 2, f_half_prev)
            inv_matrix = inv(np.identity(2) - A_half * dt / 4)
            f_half = inv_matrix @ ((np.identity(2) + A_half * dt / 4) @ f_prev)
            if norm(f_half - f_half_prev) / norm(f_half) < tol:
                break

        A_next = A_matrix(t[i], f_half)
        inv_matrix = inv(3 * np.identity(2) - A_next * dt)
        f_next = inv_matrix @ (4 * f_half - f_prev)
        f = f_next
        f_values.append(f)

    return np.array(f_values)

# Time range for t0 = 0.001 s with different time steps
timesteps = [5e-5, 2.5e-5, 1.25e-5]
solutions_lagged = {}
solutions_picard = {}

for dt in timesteps:
    t = np.arange(0, 0.001 + dt, dt)
    solutions_lagged[dt] = lagged_TBDF2(t, dt)
    solutions_picard[dt] = picard_TBDF2(t, dt)

# Assuming that we are interested in the solution at t = 0.001 s, we extract these
N_at_t0_lagged = {dt: sol[-1, 0] for dt, sol in solutions_lagged.items()}
N_at_t0_picard = {dt: sol[-1, 0] for dt, sol in solutions_picard.items()}

# Print the solutions for N at t = 0.001 s for both methods and all time steps
print("Lagged Trapezoidal/BDF-2 Scheme:")
for dt, N in N_at_t0_lagged.items():
    print(f"Delta t = {dt:.1e}, N = {N:.6f}")

print("\nFully-Iterated Picard Trapezoidal/BDF-2 Scheme:")
for dt, N in N_at_t0_picard.items():
    print(f"Delta t = {dt:.1e}, N = {N:.6f}")

# Calculate the convergence ratio and the estimated order of accuracy
def convergence_order(N_values):
    ratios = []
    orders = []
    dt_keys = sorted(N_values.keys())
    for i in range(len(dt_keys)-2):
        N1, N2, N3 = N_values[dt_keys[i]], N_values[dt_keys[i+1]], N_values[dt_keys[i+2]]
        ratio = (N1 - N2) / (N2 - N3)
        order = np.log(ratio) / np.log(2)
        ratios.append(ratio)
        orders.append(order)
    return ratios, orders

ratios_lagged, orders_lagged = convergence_order(N_at_t0_lagged)
ratios_picard, orders_picard = convergence_order(N_at_t0_picard)

print("\nConvergence Ratios and Orders of Accuracy for Lagged Scheme:")
for dt, ratio, order in zip(timesteps[:-2], ratios_lagged, orders_lagged):
    print(f"For Delta t = {dt:.1e}, Ratio: {ratio:.2f}, Order: {order:.2f}")

print("\nConvergence Ratios and Orders of Accuracy for Picard Scheme:")
for dt, ratio, order in zip(timesteps[:-2], ratios_picard, orders_picard):
    print(f"For Delta t = {dt:.1e}, Ratio: {ratio:.2f}, Order: {order:.2f}")

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Constants given in the problem
beta = 0.0075
lambda_ = 0.08
Lambda_ = 6e-5
N0 = 1
C0 = beta * N0 / (Lambda_ * lambda_)
alpha_with_feedback = 2.0 * beta
alpha_no_feedback = 0.0
Nref = 1.0
rho0_with_feedback = -6.67 * beta + alpha_with_feedback * N0 / Nref
rho0_no_feedback = -6.67 * beta

# Reactivity function with and without feedback
def reactivity_with_feedback(t, N):
    return rho0_with_feedback - alpha_with_feedback * N / Nref

def reactivity_no_feedback(t, N):
    return rho0_no_feedback

# Define the system of equations
def point_kinetics(N, C, rho):
    dN_dt = (rho - beta) / Lambda_ * N + lambda_ * C
    dC_dt = beta / Lambda_ * N - lambda_ * C
    return np.array([dN_dt, dC_dt])

# Fully-iterated Picard TBDF-2 scheme
def picard_TBDF2(t, dt, reactivity_func, tol=1e-8, max_iter=100):
    # Initial conditions
    f = np.array([N0, C0])
    # Storing the solution
    f_values = [f]

    for i in range(1, len(t)):
        f_prev = f_values[-1]
        f_half = f_prev
        for _ in range(max(2, max_iter)):
            f_half_prev = f_half
            rho = reactivity_func(t[i] - dt / 2, f_half_prev[0])
            A_half = A_matrix(t[i] - dt / 2, f_half_prev, rho)
            inv_matrix = inv(np.identity(2) - A_half * dt / 4)
            f_half = inv_matrix @ ((np.identity(2) + A_half * dt / 4) @ f_prev)

            if norm(f_half - f_half_prev) / norm(f_half) < tol:
                break

        rho = reactivity_func(t[i], f_half[0])
        A_next = A_matrix(t[i], f_half, rho)
        inv_matrix = inv(3 * np.identity(2) - A_next * dt)
        f_next = inv_matrix @ (4 * f_half - f_prev)

        f = f_next
        f_values.append(f)

    return np.array(f_values)

# A matrix function updated to include reactivity
def A_matrix(t, f, rho):
    return np.array([[(rho - beta) / Lambda_, lambda_], [beta / Lambda_, -lambda_]])

# The norm function as defined in the problem
def norm(f):
    return np.sqrt(f[0] ** 2 + f[1] ** 2)

# Time step and range for the solution
dt = 5e-5
t = np.arange(0, 0.001 + dt, dt)

# Solve the problem with and without feedback
solution_no_feedback = picard_TBDF2(t, dt, reactivity_no_feedback)
solution_with_feedback = picard_TBDF2(t, dt, reactivity_with_feedback)

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(t, solution_no_feedback[:, 0], label='No Feedback', marker='o')
plt.plot(t, solution_with_feedback[:, 0], label='With Feedback', marker='x')
plt.xlabel('Time (s)')
plt.ylabel('Neutron Density')
plt.title('Comparison of Neutron Density with and without Feedback')
plt.legend()
plt.grid(True)
plt.show()
