import numpy as np
from scipy.optimize import linprog
from simplex import simplex_iteration
import time

# FILE: test_simplex.py

import matplotlib.pyplot as plt

def generate_solvable_lp(num_random_constraints, n):
    """
    Generates a solvable LP problem.
    The domain is forced within [0, 5]^n.
    Then adds random constraints that include a known feasible x_feasible.
    """
    while True:
        # 1) Pick a feasible point in [0, 5]^n
        x_feasible = np.random.uniform(0, 5, n)

        # 2) Build bounding constraints x_i <= 5 and x_i >= 0
        A_bounds = []
        b_bounds = []
        for i in range(n):
            row_le = np.zeros(n)
            row_le[i] = 1
            A_bounds.append(row_le)
            b_bounds.append(5)

            row_ge = np.zeros(n)
            row_ge[i] = -1
            A_bounds.append(row_ge)
            b_bounds.append(0)

        A_bounds = np.array(A_bounds, dtype=float)
        b_bounds = np.array(b_bounds, dtype=float)

        # 3) Add random constraints consistent with x_feasible
        A_rand = np.random.uniform(-1, 1, (num_random_constraints, n))
        b_rand = A_rand.dot(x_feasible) + np.random.uniform(1, 5, num_random_constraints)

        A_final = np.vstack((A_bounds, A_rand))
        b_final = np.concatenate((b_bounds, b_rand))

        # 4) Random objective function
        c = np.random.uniform(-10, 10, n)

        # Use linprog to check if the problem is solvable
        res = linprog(c, A_ub=A_final, b_ub=b_final, method='highs')
        if res.success:
            return c, A_final, b_final

def test_simplex_iteration():
    for _ in range(10):
        c, A, b = generate_solvable_lp(5, 3)
        basis = list(range(A.shape[1]))  # Initialize basis with the first n indices
        x_opt, status = simplex_iteration(c, A, b, basis)
        assert x_opt is not None, f"Failed to find solution: {status}"
        assert status.startswith("Optimal solution"), f"Unexpected status: {status}"

def test_unbounded_lp():
    c = np.array([-1, -1], dtype=float)
    A = np.array([[1, 0], [0, 1]], dtype=float)
    b = np.array([1, 1], dtype=float)
    basis = [0, 1]
    x_opt, status = simplex_iteration(c, A, b, basis)
    assert x_opt is None, "Expected unbounded solution"
    assert status == "Unbounded", f"Unexpected status: {status}"

def test_infeasible_lp():
    c = np.array([1, 1], dtype=float)
    A = np.array([[1, 1], [-1, -1]], dtype=float)
    b = np.array([1, -3], dtype=float)
    basis = [0, 1]
    x_opt, status = simplex_iteration(c, A, b, basis)
    assert x_opt is None, "Expected infeasible solution"
    assert status == "Infeasible domain (singular basis)", f"Unexpected status: {status}"

def measure_performance():
    problem_sizes = [2, 5, 10, 20, 50, 100]
    mean_times = []
    std_times = []

    for size in problem_sizes:
        times = []
        for _ in range(100):
            try:
                c, A, b = generate_solvable_lp(size, size)
                basis = list(range(A.shape[1]))
                start_time = time.time()
                x_opt, status = simplex_iteration(c, A, b, basis)
                if status.startswith("Optimal solution"):
                    times.append(time.time() - start_time)
            except:
                continue

        if times:
            mean_times.append(np.mean(times))
            std_times.append(np.std(times))
        else:
            mean_times.append(float('nan'))
            std_times.append(float('nan'))

    plt.errorbar(problem_sizes, mean_times, yerr=std_times, fmt='-o')
    plt.xlabel('Problem Size (n)')
    plt.ylabel('Mean Solve Time (s)')
    plt.title('Solve Time vs Problem Size')
    plt.show()

if __name__ == "__main__":
    test_simplex_iteration()
    test_unbounded_lp()
    test_infeasible_lp()
    measure_performance()
    print("All tests passed.")