import numpy as np
import time
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from simplex import solve_lp_lu, to_standard_form, simplex_iteration_lu, simplex_iteration_straight
from test import generate_solvable_lp_linprog

def report_solve_time():
    sizes = [(50, 50), (60 , 60), (70, 70), (80, 80)]
    mean_times = []
    std_devs = []
    linprog_mean_times = []
    linprog_std_devs = []

    for m, n in sizes:
        times = []
        linprog_times = []
        for _ in range(20):
            c, A, b = generate_solvable_lp_linprog(m, n)
            basis, c_std, A_std, b_std = to_standard_form(c, A, b)
            x_opt, obj_val = simplex_iteration_straight(c_std, A_std, b_std, basis)

            try:
                start_time = time.time()
                simplex_iteration_lu(c_std, A_std, b_std, basis)
                end_time = time.time()
                elapsed_time = end_time - start_time
                times.append(elapsed_time)
            except Exception:
                continue

            try:
                start_time = time.time()
                linprog(c, A_ub=A, b_ub=b, method='simplex')
                end_time = time.time()
                elapsed_time = end_time - start_time
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