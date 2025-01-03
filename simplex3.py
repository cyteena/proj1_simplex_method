# ...existing code...
import numpy as np
import time

def convert_to_standard_form(A, b, c):
    """
    模块0：将 LP 问题转换成 Ax=b, x>=0 的标准形式
    """
    # 这里假设所有约束都可以转化为等式
    # 若原问题有不等式，可补充松弛/剩余变量处理
    m, n = A.shape
    # 假设所有约束都是 <= 形式
    A_eq = np.hstack((A, np.eye(m)))
    c_eq = np.hstack((c, np.zeros(m)))
    return A_eq.copy(), b.copy(), c_eq.copy()

def remove_redundant_constraints(A, b):
    """
    模块1：行满秩检查并移除冗余约束
    """
    # 采用QR或行阶梯判断秩
    rank = np.linalg.matrix_rank(A)
    if rank < A.shape[0]:
        # 简要示例：保留前 rank 行（假定其为独立行）
        A = A[:rank,:]
        b = b[:rank]
    return A.copy(), b.copy()

def phase1(A, b):
    """
    两阶段法第一阶段：目标函数 min sum(ai)，ai 为人工变量
    """
    m, n = A.shape
    # 人工变量
    A_phase1 = np.hstack((A, np.eye(m)))
    c_phase1 = np.zeros(n+m)
    c_phase1[n:] = 1.0
    basis = list(range(n, n+m))
    # x 初值
    x0 = np.zeros(n+m)
    x0[basis] = b
    # 进入单纯形
    x_sol, val = simplex_iteration(A_phase1, b, c_phase1, basis, two_phase=True)
    return x_sol, val, basis, A_phase1, c_phase1

def phase2(A, b, c, x_sol, basis, A_phase1):
    """
    两阶段法第二阶段
    """
    # 删除人工变量列
    m, n = A.shape
    # 若第一阶段可行
    # 恢复原始列并进入单纯形
    x_sol = x_sol[:n]
    new_basis = [i for i in basis if i < n]
    x_sol, obj_val = simplex_iteration(A, b, c, new_basis)
    return x_sol, obj_val

def initialize_feasible_solution(A, b, c):
    """
    模块2：使用两阶段法构造可行基解
    """
    x_sol, val, basis, A_phase1, c_phase1 = phase1(A, b)
    # 判断是否可行
    if val > 1e-8:
        return None, None
    x_sol, obj_val = phase2(A, b, c, x_sol, basis, A_phase1)
    return x_sol, obj_val

def simplex_iteration(A, b, c, basis, two_phase=False):
    """
    模块3：单纯形法迭代
    A: shape(m,n)
    b: shape(m,)
    c: shape(n,)
    basis: 基变量索引列表
    """
    m, n = A.shape
    # 基本可行解
    B = A[:, basis]
    invB = np.linalg.inv(B)
    x = np.zeros(n)
    x[basis] = invB.dot(b)
    
    while True:
        # 计算检验数
        cb = c[basis]
        pi = cb @ invB
        reduced_costs = c - pi @ A
        # 寻找正的检验数进入基
        entering = None
        for j in range(n):
            if reduced_costs[j] > 1e-9:
                entering = j
                break
        if entering is None:
            # 最优
            return x, cb @ x[basis]
        # 计算离基变量
        direction = invB.dot(A[:, entering])
        if all(direction <= 1e-9):
            # 无界
            return None, None
        # ratio test
        ratios = []
        for i, d in enumerate(direction):
            if d > 1e-9:
                ratios.append(x[basis[i]]/d)
            else:
                ratios.append(np.inf)
        leaving_index = np.argmin(ratios)
        leaving = basis[leaving_index]
        # pivot
        basis[leaving_index] = entering
        # 更新 B, invB
        E = np.eye(m)
        E[:, leaving_index] = -direction/direction[leaving_index]
        E[leaving_index, leaving_index] = 1.0/direction[leaving_index]
        invB = E @ invB
        x[basis] = 0
        x[basis] = invB.dot(b)

def main():
    np.random.seed(0)
    sizes = [5, 10]  # 可根据需求扩展
    for size in sizes:
        timings = []
        for _ in range(20):
            A = np.random.rand(size, size)
            b = np.random.rand(size)
            c = np.random.rand(size)
            A_std, b_std, c_std = convert_to_standard_form(A, b, c)
            A_std, b_std = remove_redundant_constraints(A_std, b_std)
            start = time.time()
            x0, obj_val = initialize_feasible_solution(A_std, b_std, c_std)
            end  = time.time()
            if x0 is not None:
                timings.append(end-start)
        if len(timings)>0:
            print(f"规模 {size}：平均时间={np.mean(timings):.6f}秒，标准差={np.std(timings):.6f}")
        else:
            print(f"规模 {size}：无可行解案例（跳过）")

if __name__ == "__main__":
    main()
# ...existing code...