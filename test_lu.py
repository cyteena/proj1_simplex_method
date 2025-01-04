import numpy as np
import os
import time
import matplotlib.pyplot as plt
from simplex import simplex_iteration_lu, simplex_iteration_straight, to_standard_form
from test import generate_solvable_lp_linprog
from scipy.optimize import linprog


def test_simplex_methods(sizes, num_tests=20):
    """
    Test the simplex_iteration_straight and simplex_iteration_lu functions on LP problems of various sizes.
    """
    results = []

    for m, n in sizes:
        success_straight = 0
        success_lu = 0
        time_straight = []
        time_lu = []

        for _ in range(num_tests):
            c, A, b = generate_solvable_lp_linprog(m, n)
            res = linprog(c, A_ub=A, b_ub=b, method='simplex')
            basis, c_std, A_std, b_std= to_standard_form(c, A, b)

            # Test simplex_iteration_straight
            start_time = time.time()
            result_straight, status_straight = simplex_iteration_straight(c_std, A_std, b_std, basis)
            end_time = time.time()
            if status_straight.startswith("Optimal solution"):
                time_straight.append(end_time - start_time)
                success_straight += 1

            # Test simplex_iteration with LU decomposition
            start_time = time.time()
            result_lu, status_lu = simplex_iteration_lu(c_std, A_std, b_std, basis)
            end_time = time.time()
            if status_lu.startswith("Optimal solution"):
                time_lu.append(end_time - start_time)
                success_lu += 1


        avg_time_straight = np.mean(time_straight) if time_straight else None
        avg_time_lu = np.mean(time_lu) if time_lu else None
        results.append((m, n, success_straight, success_lu, avg_time_straight, avg_time_lu))

    return results

def plot_results(size, success_straight, success_lu, num_tests):
    m, n = size
    labels = ['simplex_iteration_straight', 'simplex_iteration_lu']
    successes = [success_straight, success_lu]
    failures = [num_tests - success_straight, num_tests - success_lu]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, successes, width, label='Successes')
    rects2 = ax.bar(x + width/2, failures, width, label='Failures')

    ax.set_ylabel('Number of Problems')
    ax.set_title(f'Comparison of Simplex Methods for {m}x{n} LP Problems')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()


    # 保存图像到other文件夹
    if not os.path.exists('other'):
        os.makedirs('other')
    filename = f'other/comparison_{m}x{n}_{num_tests}_tests.png'
    plt.savefig(filename)

    plt.show()

if __name__ == "__main__":
    num_tests = 30
    sizes = [(20, 40), (30, 35)] 
    results = test_simplex_methods(sizes, num_tests)
    for m, n, success_straight, success_lu, avg_time_straight, avg_time_lu in results:
        print(f"Size {m}x{n}:")
        print(f"  simplex_iteration_straight solved {success_straight} out of {num_tests} problems")
        print(f"  simplex_iteration_lu solved {success_lu} out of {num_tests} problems")
        plot_results((m, n), success_straight, success_lu, num_tests)