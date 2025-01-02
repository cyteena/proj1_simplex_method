import numpy as np
import time
import matplotlib.pyplot as plt

def generate_random_lp(m, n):
    """
    Generate a random LP problem.
    """
    c = np.random.uniform(-10, 10, n)
    A = np.random.uniform(-10, 10, (m, n))
    b = np.random.uniform(0, 10, m)
    return c, A, b

def report_solve_time():
    sizes = [(5, 10), (10, 20), (20, 40), (40, 80), (80, 160)]
    mean_times = []
    std_devs = []

    for m, n in sizes:
        times = []
        for _ in range(20):
            c, A, b = generate_random_lp(m, n)
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
    plt.xlabel('Problem Size (m, n)')
    plt.ylabel('Mean Solve Time (s)')
    plt.title('Simplex Method Solve Time vs Problem Size')
    plt.show()

if __name__ == "__main__":
    report_solve_time()