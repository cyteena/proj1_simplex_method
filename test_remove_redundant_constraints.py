import numpy as np
from simplex import remove_redundant_constraints

def test_remove_redundant_constraints():
    # 测试案例1：无冗余约束
    A1 = np.array([[1, 2], [3, 4]], dtype=float)
    b1 = np.array([5, 6], dtype=float)
    A1_filtered, b1_filtered = remove_redundant_constraints(A1, b1)
    print("Test case 1:")
    print("A1_filtered:\n", A1_filtered)
    print("b1_filtered:\n", b1_filtered)
    assert np.allclose(A1_filtered, A1), "Test case 1 failed"
    assert np.allclose(b1_filtered, b1), "Test case 1 failed"

    # 测试案例2：有冗余约束
    A2 = np.array([[1, 2], [2, 4]], dtype=float)
    b2 = np.array([5, 10], dtype=float)
    A2_filtered, b2_filtered = remove_redundant_constraints(A2, b2)
    expected_A2_filtered = np.array([[1, 2]], dtype=float)
    expected_b2_filtered = np.array([5], dtype=float)
    print("Test case 2:")
    print("A2_filtered:\n", A2_filtered)
    print("b2_filtered:\n", b2_filtered)
    assert np.allclose(A2_filtered, expected_A2_filtered), "Test case 2 failed"
    assert np.allclose(b2_filtered, expected_b2_filtered), "Test case 2 failed"

    # 测试案例3：部分冗余约束
    A3 = np.array([[1, 2, 3], [2, 4, 6], [1, 1, 1]], dtype=float)
    b3 = np.array([6, 12, 3], dtype=float)
    A3_filtered, b3_filtered = remove_redundant_constraints(A3, b3)
    expected_A3_filtered = np.array([[1, 2, 3], [1, 1, 1]], dtype=float)
    expected_b3_filtered = np.array([6, 3], dtype=float)
    print("Test case 3:")
    print("A3_filtered:\n", A3_filtered)
    print("b3_filtered:\n", b3_filtered)
    assert np.allclose(A3_filtered, expected_A3_filtered), "Test case 3 failed"
    assert np.allclose(b3_filtered, expected_b3_filtered), "Test case 3 failed"

    # 测试案例4：随机生成的矩阵
    np.random.seed(0)
    A4 = np.random.rand(10, 5)
    b4 = np.random.rand(10)
    A4_filtered, b4_filtered = remove_redundant_constraints(A4, b4)
    print("Test case 4:")
    print("A4_filtered shape:", A4_filtered.shape)
    print("b4_filtered shape:", b4_filtered.shape)
    assert A4_filtered.shape[0] <= A4.shape[0], "Test case 4 failed"
    assert A4_filtered.shape[1] == A4.shape[1], "Test case 4 failed"
    assert b4_filtered.shape[0] == A4_filtered.shape[0], "Test case 4 failed"

    print("All test cases passed!")

if __name__ == "__main__":
    test_remove_redundant_constraints()