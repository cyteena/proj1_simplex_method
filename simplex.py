import numpy as np

def to_standard_form(c, A, b):
    # ...existing code...
    # Convert ≤ constraints to standard form by adding slack variables
    m, n = A.shape
    I = np.eye(m)
    A_std = np.hstack((A, I))
    c_std = np.concatenate((c, np.zeros(m)))
    b_std = b.copy()
    return c_std, A_std, b_std

def check_slack_variables(x_opt, n):
    """
    Check if the slack variables are all non-negative to determine feasibility.
    """
    slack_variables = x_opt[n:]
    if all(slack_variables >= 0):
        return True, "Feasible domain"
    else:
        return False, "Infeasible domain"
    

def remove_redundant_constraints(A, b, tol=1e-9):
    """
    A pivot-based approach to remove redundant constraints.
    This avoids using SVD for large problems and can be faster in practice.
    """
    # Make copies to avoid modifying original arrays
    A_work = A.copy().astype(float)
    b_work = b.copy().astype(float)

    m, n = A_work.shape
    row = 0

    # Perform a straightforward pivot-based row-reduction
    for col in range(n):
        # 1) Find pivot
        pivot = row
        while pivot < m and abs(A_work[pivot, col]) < tol:
            pivot += 1
        if pivot == m:
            continue  # No pivot in this column

        # 2) Swap pivot row to current row if needed
        if pivot != row:
            A_work[[row, pivot]] = A_work[[pivot, row]]
            b_work[[row, pivot]] = b_work[[pivot, row]]

        # 3) Normalize pivot row
        pivot_val = A_work[row, col]
        if abs(pivot_val) < tol:
            continue
        A_work[row] /= pivot_val
        b_work[row] /= pivot_val

        # 4) Eliminate below pivot
        for r in range(row + 1, m):
            factor = A_work[r, col]
            A_work[r] -= factor * A_work[row]
            b_work[r] -= factor * b_work[row]

        row += 1
        if row == m:
            break

    # Identify nonzero rows
    nonzero_idx = []
    for i in range(m):
        # If row is effectively not zero
        if not all(abs(A_work[i]) < tol) or abs(b_work[i]) >= tol:
            nonzero_idx.append(i)

    # Extract the filtered constraints
    A_filtered = A_work[nonzero_idx, :]
    b_filtered = b_work[nonzero_idx]

    return A_filtered, b_filtered



def big_m_method(c_std, A_std, b_std):
    m, n = A_std.shape
    A_art = np.hstack((A_std, np.eye(m)))
    big_M = 1e6
    c_art = np.concatenate((c_std, [big_M] * m))
    basis = list(range(n, n+m))
    x_init = np.zeros(n+m)  # 修正维度
    x_init[basis] = b_std
    return basis, x_init, c_art, A_art

def simplex_iteration(c, A, b, basis):
    """
    Simplex iteration that distinguishes two types of 'no solution':
    1) No feasible region
    2) Unbounded problem
    """
    max_iter = 50
    x = np.zeros(A.shape[1])
    if len(basis) > len(b):
        return None, "Infeasible domain (basis size mismatch)"
    x[basis] = b[:len(basis)]

    for _ in range(max_iter):
        # Compute reduced costs
        cb = c[basis]
        B = A[:, basis]
        try:
            invB = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            return None, "Infeasible domain (singular basis)"
        lambd = cb @ invB
        r = c - lambd @ A

        # If all reduced costs >= 0, we have an optimal solution
        if all(r >= 0):
            return x, "Optimal solution = " + str(c @ x)

        # Choose entering variable
        entering = np.argmin(r)

        # Determine direction
        d = invB @ A[:, entering]

        # If direction is non-positive => unbounded
        if all(d <= 0):
            return None, "Unbounded"

        # Ratio test using Bland's rule (smallest subscript rule)
        ratios = []
        for i in range(len(basis)):
            if d[i] > 0:
                ratios.append((x[basis][i] / d[i], i))
        if not ratios:
            # No positive direction => unbounded
            return None, "Unbounded"

        leaving = min(ratios, key=lambda x: x[0])[1]
        basis[leaving] = entering

        # Update solution
        x = np.zeros(A.shape[1])
        x[basis] = np.linalg.inv(A[:, basis]) @ b

    # If max_iter exceeded, treat as infeasible or stuck
    return None, "Infeasible or iteration limit reached"

def solve_lp(c, A, b):
    # 输入验证
    if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray) or not isinstance(c, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    
    if A.shape[0] != len(b):
        raise ValueError("Inconsistent dimensions between A and b")
    
    if A.shape[1] != len(c):
        raise ValueError("Inconsistent dimensions between A and c")
        
    c_std, A_std, b_std = to_standard_form(c, A, b)
    basis, x_init, c_art, A_art = big_m_method(c_std, A_std, b_std)
    
    try:
        # 修正参数顺序，确保维度匹配
        x_opt, obj_val = simplex_iteration(A=A_art, b=b_std, c=c_art, basis=basis)
    except np.linalg.LinAlgError:
        return None, "Numerical instability encountered"
    
    if x_opt is None:
        return None, "Unbounded solution"
    
    # 只返回原始变量的解
    x_original = x_opt[:len(c)]
    
    # 计算正确的目标函数值（使用原始目标函数系数）
    obj_val = c @ x_original
    
    # 检查可行性
    feasible, status = check_slack_variables(x_opt, len(c))
    if not feasible:
        return None, status
    
    return x_original, obj_val
    

def check_degeneracy(x):
    # ...existing code...
    # Simple placeholder: check if any x is nearly zero
    eps = 1e-12
    is_degenerate = any(abs(val) < eps for val in x)
    return is_degenerate

if __name__ == "__main__":
    # Example usage
    c = np.array([-2, -3, 5], dtype=float)
    A = np.array([[2, -1, 1],
                  [1, 1, 3],
                  [1, -1, 4],
                  [3, 1, 2]], dtype=float)
    b = np.array([2, 5, 6, 8], dtype=float)
    
    x_opt, obj_val = solve_lp(c=c, A=A, b=b)
    if isinstance(obj_val, str):
        print(f"Solution: {x_opt}")
        print(f"Status: {obj_val}")
    else:
        print(f"Solution: {x_opt}")
        print(f"Objective value: {obj_val}")
    if x_opt is not None:
        print("Degenerate:", check_degeneracy(x_opt))
    else:
        print("Degenerate: N/A")


