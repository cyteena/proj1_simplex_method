import json
import numpy as np
from scipy.optimize import linprog
from simplex import solve_lp_lu

def read_failed_examples(filename="failed_example.json"):
    with open(filename, "r") as f:
        examples = json.load(f)
    return examples

def solve_example(c, A, b):
    # 使用自定义的 solve_lp 函数
    x_opt, solve_lp_status = solve_lp_lu(np.array(c), np.array(A), np.array(b))
    
    # 使用 scipy.optimize.linprog 函数
    res = linprog(c, A_ub=A, b_ub=b, method='simplex')
    linprog_status = "Optimal" if res.success else "Infeasible or Unbounded"
    
    return x_opt, solve_lp_status, res.x, linprog_status

if __name__ == "__main__":
    examples = read_failed_examples()
    
    if examples:
        # 读取第一组 c, A, b
        example = examples[0]
        c = example["c"]
        A = example["A"]
        b = example["b"]
        
        # 求解
        x_opt, solve_lp_status, linprog_x, linprog_status = solve_example(c, A, b)
        
        # 输出结果
        print("solve_lp solution:", x_opt)
        print("solve_lp status:", solve_lp_status)
        print("linprog solution:", linprog_x)
        print("linprog status:", linprog_status)
    else:
        print("No examples found in failed_example.json")