import json
import numpy as np
from scipy.optimize import linprog
from simplex import solve_lp_lu
from test import generate_solvable_lp_linprog
from matplotlib.colors import LinearSegmentedColormap
from simplex import to_standard_form, simplex_iteration_lu, simplex_iteration_straight, big_m_method
import matplotlib.pyplot as plt

def compare_with_scipy(c, A, b):
    # 使用 solve_lp 求解

    basis, c_std, A_std, b_std = to_standard_form(c, A, b)

    x_opt, solve_lp_status = simplex_iteration_straight(c_std, A_std, b_std, basis)
    x_opt, solve_lp_status = simplex_iteration_lu(c_std, A_std, b_std, basis)
    # 使用 scipy.optimize.linprog 求解
    res = linprog(c, A_ub=A, b_ub=b, method='simplex')
    
    if x_opt is None:
        solve_lp_status = solve_lp_status
    else:
        solve_lp_status = "Optimal"
    
    if res.success:
        linprog_status = "Optimal"
    else:
        linprog_status = res.message
    
    # 比较结果
    if solve_lp_status == linprog_status:
        if solve_lp_status == "Optimal":
            # 比较解的值
            if np.allclose(x_opt[:len(c)], res.x, atol=1e-5):
                return True, "Both methods found the same optimal solution."
            else:
                return False, "Both methods found different solutions."
        else:
            return True, f"Both methods returned the same status: {solve_lp_status}."
    else:
        return False, f"Different statuses: solve_lp returned {solve_lp_status}, linprog returned {linprog_status}."

def test_compare_with_scipy(num_tests=20):
    sizes = [(30, 35), (40, 40), (50, 50), (60, 60)]
    results = []
    success_rates = []

    for m, n in sizes:
        success_count = 0
        for _ in range(num_tests):
            c, A, b = generate_solvable_lp_linprog(m, n)
            result, message = compare_with_scipy(c, A, b)
            results.append((result, message, c, A, b))
            if result:
                success_count += 1
            print(f"Size: {(m, n)}, Result: {result}, Message: {message}")
        success_rate = success_count / num_tests
        success_rates.append((m, n, success_rate))
        print(f"Size: {(m, n)}, Success rate: {success_rate}")

    return results, success_rates

def plot_success_rate(success_rates):
    sizes = [f"{m}x{n}" for m, n, _ in success_rates]
    success_rates_values = [success_rate for _, _, success_rate in success_rates]

    x = np.arange(len(sizes))

    # 创建颜色渐变
    cmap = LinearSegmentedColormap.from_list("custom_gradient", ["#D8BFD8", "#FFB6C1"])
    colors = cmap(np.linspace(0, 1, len(success_rates_values)))

    fig, ax = plt.subplots()
    bars = ax.bar(x, success_rates_values, color=colors)

    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate of solve_lp_lu by Size')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)

    for i, v in enumerate(success_rates_values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')

    fig.tight_layout()
    plt.show()
    
def write_failed_examples_to_json(test_results, filename="failed_example.json"):
    failed_examples = []
    for result, message, c, A, b in test_results:
        if not result and "solve_lp returned Unbounded solution, linprog returned Optimal" in message:
            failed_examples.append({
                "c": c.tolist(),
                "A": A.tolist(),
                "b": b.tolist()
            })
    with open(filename, "w") as f:
        json.dump(failed_examples, f, indent=4)


if __name__ == "__main__":

    test_results, success_rates = test_compare_with_scipy()
    plot_success_rate(success_rates)
    write_failed_examples_to_json(test_results)
    


    
