import numpy as np
import matplotlib.pyplot as plt
from simplex import simplex_iteration_lu, simplex_iteration_straight, to_standard_form

def generate_random_lp(m, n):
    """
    Generate a random LP problem with integer coefficients.
    """
    c = np.random.randint(-10, 11, n)
    A = np.random.randint(-10, 11, (m, n))
    b = np.random.randint(0, 11, m)
    return c, A, b

def test_simplex_methods(num_tests=100):
    m, n = 20, 20  # Dimensions of the LP problems
    success_straight = 0
    success_lu = 0

    for _ in range(num_tests):
        c, A, b = generate_random_lp(m, n)
        basis, c, A, b = to_standard_form(c, A, b)

        # Test simplex_iteration_straight
        result_straight, status_straight = simplex_iteration_straight(c, A, b, basis)
        if status_straight.startswith("Optimal solution"):
            success_straight += 1

        # Test simplex_iteration with LU decomposition
        result_lu, status_lu = simplex_iteration_lu(c, A, b, basis)
        if status_lu.startswith("Optimal solution"):
            success_lu += 1

    return success_straight, success_lu

def plot_results(success_straight, success_lu, num_tests):
    labels = ['simplex_iteration_straight', 'simplex_iteration']
    successes = [success_straight, success_lu]
    failures = [num_tests - success_straight, num_tests - success_lu]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, successes, width, label='Successes')
    rects2 = ax.bar(x + width/2, failures, width, label='Failures')

    ax.set_ylabel('Number of Problems')
    ax.set_title('Comparison of Simplex Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    num_tests = 100
    success_straight, success_lu = test_simplex_methods(num_tests)
    print(f"simplex_iteration_straight solved {success_straight} out of {num_tests} problems.")
    print(f"simplex_iteration with LU decomposition solved {success_lu} out of {num_tests} problems.")
    plot_results(success_straight, success_lu, num_tests)