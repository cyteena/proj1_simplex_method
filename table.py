import numpy as np
from scipy.optimize import linprog

def simplex_big_M(f, A, b):
    # 输入维度的检查
    m, n = A.shape  # A是m*n维的
    M = 1e6  # 大M的值

    # 初始化单纯形表
    S = np.hstack([A, np.eye(m), b.reshape(-1, 1)])
    c = np.hstack([f, -M * np.ones(m), 0])
    S = np.vstack([S, c])

    # 单纯形迭代
    while np.any(S[-1, :-1] < 0):
        # 选择入基变量
        col = np.argmin(S[-1, :-1])
        if np.all(S[:-1, col] <= 0):
            raise ValueError("Linear program is unbounded.")
        
        # 选择出基变量
        ratios = np.divide(S[:-1, -1], S[:-1, col], out=np.full_like(S[:-1, -1], np.inf), where=S[:-1, col] > 0)
        row = np.argmin(ratios)

        # 枢轴操作
        S[row, :] /= S[row, col]
        for i in range(S.shape[0]):
            if i != row:
                S[i, :] -= S[i, col] * S[row, :]

    # 解析解
    x = np.zeros(n)
    for i in range(n):
        # 找到基变量
        j = np.where(S[:, i] == 1)[0]
        k = np.where(S[:, i] == 0)[0]
        if len(j) == 1 and len(k) == m:
            # i为基本元列号，j是行号
            x[i] = S[j[0], -1]

    y = S[-1, -1]  # 最优解
    return x, y

# Example usage
if __name__ == "__main__":
    f = np.array([-2, -3, 5], dtype=float)
    A = np.array([[2, -1, 1],
                  [1, 1, 3],
                  [1, -1, 4],
                  [3, 1, 2]], dtype=float)
    b = np.array([2, 5, 6, 8], dtype=float)
    
    x, y = simplex_big_M(f, A, b)
    res = linprog(f, A_ub=A, b_ub=b, method='simplex')
    print(res.x)
    print("Optimal solution x:", x)
    print("Objective value y:", y)