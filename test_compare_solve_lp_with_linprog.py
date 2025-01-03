import numpy as np
from scipy.optimize import linprog
from simplex import solve_lp
from test import generate_solvable_lp

def compare_with_scipy(c, A, b):
    # 使用 solve_lp 求解
    x_opt, obj_val = solve_lp(c, A, b)
    
    # 使用 scipy.optimize.linprog 求解
    res = linprog(c, A_ub=A, b_ub=b, method='simplex')
    
    if x_opt is None:
        solve_lp_status = obj_val
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

def test_compare_with_scipy():
    sizes = [(5, 5), (5, 6), (7, 8), (9, 9), (10, 12), (12, 15)]
    results = []

    for m, n in sizes:
        for _ in range(20):
            c, A, b = generate_solvable_lp(m, n)
            result, message = compare_with_scipy(c, A, b)
            results.append((result, message))
            print(f"Size: {(m, n)}, Result: {result}, Message: {message}")

    return results    
    

if __name__ == "__main__":
    # # 示例使用
    # c = np.array([-2, -3, 5], dtype=float)
    # A = np.array([[2, -1, 1],
    #             [1, 1, 3],
    #             [1, -1, 4],
    #             [3, 1, 2]], dtype=float)
    # b = np.array([2, 5, 6, 8], dtype=float)

    # result, message = compare_with_scipy(c, A, b)
    # print(message)

    test_results = test_compare_with_scipy()
    # 统计结果
    # success_count = sum(1 for result, _ in test_results if result)
    # total_tests = len(test_results)
    # print(f"Total tests: {total_tests}, Successes: {success_count}, Failures: {total_tests - success_count}")


    
