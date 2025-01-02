import numpy as np
from scipy.optimize import linprog

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

# 示例使用
c = np.array([-2, -3, 5], dtype=float)
A = np.array([[2, -1, 1],
              [1, 1, 3],
              [1, -1, 4],
              [3, 1, 2]], dtype=float)
b = np.array([2, 5, 6, 8], dtype=float)

result, message = compare_with_scipy(c, A, b)
print(message)