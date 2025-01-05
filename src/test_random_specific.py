import numpy as np
import time
import matplotlib.pyplot as plt
from simplex import solve_lp_lu, to_standard_form, simplex_iteration_straight, simplex_iteration_lu, remove_redundant_constraints
from scipy.optimize import linprog
from test import generate_solvable_lp_linprog

def generate_random_lp(m, n):
    """
    Generate a random LP problem.
    """
    c = np.random.uniform(-10, 10, n)
    A = np.random.uniform(-10, 10, (m, n))
    b = np.random.uniform(0, 10, m)
    return c, A, b

def test_random_cases():
    sizes = [(20, 20), (20, 30), (30, 35), (30, 40)]
    mean_times = []
    std_devs = []

    for m, n in sizes:
        times = []
        for _ in range(20):
            c, A, b = generate_solvable_lp_linprog(m, n)
            basis, c_std, A_std, b_std = to_standard_form(c, A, b)
            try:
                start_time = time.time()
                x_opt, obj_val = simplex_iteration_lu(c_std, A_std, b_std, basis=basis)
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
    plt.xlabel('Problem Size (m, n)')
    plt.ylabel('Mean Solve Time (s)')
    plt.title('Simplex Method Solve Time vs Problem Size')
    plt.show()

def test_specific_cases():
    # 正常求解
    c1 = np.array([-2, -3, 5], dtype=float)
    A1 = np.array([[2, -1, 1],
                   [1, 1, 3],
                   [1, -1, 4],
                   [3, 1, 2]], dtype=float)
    b1 = np.array([2, 5, 6, 8], dtype=float)
    x_opt, obj_val = solve_lp_lu(c1, A1, b1)
    print("正常求解:")
    print("Optimal solution:", x_opt)
    print("Optimal objective value:", obj_val)

    # 有冗余约束
    c2 = np.array([1, 1], dtype=float)
    A2 = np.array([[1, 2], [2, 4]], dtype=float)
    b2 = np.array([5, 10], dtype=float)
    A2_filtered, b2_filtered = remove_redundant_constraints(A2, b2)
    print("使用linprog求解")
    print(linprog(c2, A_ub=A2, b_ub=b2))
    x_opt, obj_val = solve_lp_lu(c2, A2_filtered, b2_filtered)
    print("有冗余约束:")
    print("Optimal solution:", x_opt)
    print("Optimal objective value:", obj_val)

    # 无可行域
    c3 = np.array([1, 1], dtype=float)
    A3 = np.array([[1, 1], [-1, -1]], dtype=float)
    b3 = np.array([1, -3], dtype=float)
    x_opt, obj_val = solve_lp_lu(c3, A3, b3)
    print("无可行域:")
    print("Optimal solution:", x_opt)
    print("Optimal objective value:", obj_val)

    # 无界
    c4 = np.array([-1, 0], dtype=float)
    A4 = np.array([[1, -1], [-1, 1]], dtype=float)
    b4 = np.array([0, 0], dtype=float)
    x_opt, obj_val = solve_lp_lu(c4, A4, b4)
    print("无界:")
    print("Optimal solution:", x_opt)
    print("Optimal objective value:", obj_val)

if __name__ == "__main__":
    test_random_cases()
    test_specific_cases()