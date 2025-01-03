import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from simplex import solve_lp

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

def test_solvable_performance():
    sizes = [(5, 5), (10, 10), (20, 20)]
    mean_times = []
    std_devs = []

    for (m, n) in sizes:
        times = []
        for _ in range(50):
            c, A, b = generate_solvable_lp(m, n)
            start_time = time.time()
            try:
                x_opt, obj_val = solve_lp(c, A, b)
                if x_opt is not None:
                    elapsed_time = time.time() - start_time
                    times.append(elapsed_time)
            except Exception:
                continue

        if times:
            mean_times.append(np.mean(times))
            std_devs.append(np.std(times))
        else:
            mean_times.append(None)
            std_devs.append(None)

    # Filter out None values
    filtered_sizes = [size for size, mean in zip(sizes, mean_times) if mean is not None]
    filtered_mean_times = [mean for mean in mean_times if mean is not None]
    filtered_std_devs = [std for std in std_devs if std is not None]

    plt.errorbar([str(size) for size in filtered_sizes], filtered_mean_times, yerr=filtered_std_devs, fmt='-o')
    plt.xlabel('Problem Size (Random Constraints, Variables)')
    plt.ylabel('Mean Solve Time (s)')
    plt.title('Simplex Method Solvable Performance')
    plt.show()

if __name__ == "__main__":
    test_solvable_performance()