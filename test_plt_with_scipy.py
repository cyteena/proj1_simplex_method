import numpy as np
import time
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from simplex import solve_lp

def generate_random_lp(m, n):
    """
    Generate a random LP problem.
    """
    c = np.random.uniform(-10, 10, n)
    A = np.random.uniform(-10, 10, (m, n))
    b = np.random.uniform(0, 10, m)
    return c, A, b

def generate_solvable_lp(m, n):
    while True:
        c, A, b = generate_random_lp(m, n)
        x_opt, obj_val = solve_lp(c, A, b)
        if x_opt is not None:
            return c, A, b

def report_solve_time():
    sizes = [(10, 20), (20, 40), (30, 60), (40, 80), (50, 100)]
    mean_times = []
    std_devs = []
    linprog_mean_times = []
    linprog_std_devs = []

    for m, n in sizes:
        times = []
        linprog_times = []
        for _ in range(20):
            A = np.random.rand(m, n)
            b = np.random.rand(m)
            c = np.random.rand(n)

            try:
                start_time = time.time()
                solve_lp(c, A, b)
                elapsed_time = time.time() - start_time
                times.append(elapsed_time)
            except Exception:
                continue

            try:
                start_time = time.time()
                linprog(c, A_eq=A, b_eq=b, method='simplex')
                elapsed_time = time.time() - start_time
                linprog_times.append(elapsed_time)
            except Exception:
                continue

        if times:
            mean_times.append(np.mean(times))
            std_devs.append(np.std(times))
        else:
            mean_times.append(None)
            std_devs.append(None)

        if linprog_times:
            linprog_mean_times.append(np.mean(linprog_times))
            linprog_std_devs.append(np.std(linprog_times))
        else:
            linprog_mean_times.append(None)
            linprog_std_devs.append(None)

    # Filter out None values
    filtered_sizes = [size for size, mean in zip(sizes, mean_times) if mean is not None]
    filtered_mean_times = [mean for mean in mean_times if mean is not None]
    filtered_std_devs = [std for std in std_devs if std is not None]
    filtered_linprog_mean_times = [mean for mean in linprog_mean_times if mean is not None]
    filtered_linprog_std_devs = [std for std in linprog_std_devs if std is not None]

    plt.errorbar([str(size) for size in filtered_sizes], filtered_mean_times, yerr=filtered_std_devs, fmt='-o', label='Custom Simplex')
    plt.errorbar([str(size) for size in filtered_sizes], filtered_linprog_mean_times, yerr=filtered_linprog_std_devs, fmt='-x', label='SciPy linprog')
    plt.xlabel('Problem Size (m, n)')
    plt.ylabel('Mean Solve Time (s)')
    plt.title('Simplex Method Solve Time vs Problem Size')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    report_solve_time()