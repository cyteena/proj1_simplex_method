# generate_input.py
# 生成一个 n x m 的随机矩阵 A 和一个 m 维的随机向量 x，
# 计算增广矩阵 [A | b]，其中 b = A * x，生成另一个 m 维向量 c，
# 并将结果输出到两个文件：LP_input.in 和 LP_expected_output.out。
# 其中 LP_input.in 存储了线性规划的标准形式，输出格式见代码中注释；
# 而 LP_expected_output.out 中存储了该 LP 问题的最优点和最优值。


import numpy as np
import random
from scipy.optimize import linprog

def generate_random_input():
    m = random.randint(10, 200) # 随机生成 m，范围在 [10, 200] 之间
    n = m - random.randint(0, m // 3) # 随机生成 n，满足 n <= m 且 m - n <= m / 3
    return n, m

def generate_matrix_and_vectors(n, m, output_file, expected_output_file):
    # 生成一个 n x m 的随机矩阵 A，矩阵元素在 [-10, 10] 范围内
    A = np.random.uniform(-10, 10, (n, m))
    # 生成一个 m 维的随机向量 x，元素在 [0, 10] 范围内
    x = np.random.uniform(0, 10, m)
    b = A @ x # 计算 b = A * x
    # 生成一个 m 维的随机向量 c，元素在 [-10, 10] 范围内
    c = np.random.uniform(-10, 10, m)
    

    # 输出格式为：
    # 第一行为 n m
    # 第二行为 c^T
    # 接下来是增广矩阵 [A | b] 的各行
    result = []
    result.append(f"{n} {m}")
    result.append(" ".join(f"{value:.10f}" for value in c))
    augmented_matrix = np.hstack((A, b.reshape(-1, 1)))
    for row in augmented_matrix:
        result.append(" ".join(f"{value:.10f}" for value in row))
    # 写入 LP_input.in 文件
    with open(output_file, "w") as f:
        f.write("\n".join(result) + "\n")
    
    # 注意上面随机生成的 x 仅是一可行解，这里再将最优解求解出来（使用 Python 包中自带求解器）
    result = linprog(c = c, A_eq = A, b_eq = b, bounds = (0, None))
    if not result.success:
        # 若求解失败，则再重新生成一遍
        return generate_matrix_and_vectors(n, m, output_file, expected_output_file)

    # 将最优值 result.fun 及最优点 result.x 输出到 LP_expected_output.out 文件
    with open(expected_output_file, "w") as f:
        f.write(f"{result.fun:.10f}\n")  # 输出最优值 result.fun
        f.write(" ".join(f"{value:.10f}" for value in result.x) + "\n")  # 输出最优点 result.x

if __name__ == "__main__":
    n, m = generate_random_input()
    print(f"生成的 n = {n}, m = {m}")
    
    input_file = "LP_input.in"
    expected_output_file = "LP_expected_output.out"
    generate_matrix_and_vectors(n, m, input_file, expected_output_file)
    print(f"结果已保存到 {input_file} 和 {expected_output_file}")
