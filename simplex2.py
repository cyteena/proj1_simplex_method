import numpy as np

def simplex(c, A, b):
    """
    Solves the linear programming problem:
    maximize    c^T x
    subject to  Ax <= b, x >= 0

    Parameters:
    c : array_like, shape (n,)
        Coefficients of the linear objective function to be maximized.
    A : array_like, shape (m, n)
        2-D array which, when matrix-multiplied by x, gives the values of the
        inequality constraints at x.
    b : array_like, shape (m,)
        1-D array of values representing the upper-bound of each inequality
        constraint (row) in A.

    Returns:
    x : array, shape (n,)
        Solution vector.
    """
    m, n = A.shape

    # Create the tableau
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:m, :n] = A
    tableau[:m, n:n + m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[-1, :n] = -c

    while True:
        # Check if we can stop
        if np.all(tableau[-1, :-1] >= 0):
            break

        # Pivot column
        pivot_col = np.argmin(tableau[-1, :-1])

        # Pivot row
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        pivot_row = np.where(ratios > 0, ratios, np.inf).argmin()

        # Pivot
        pivot = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    # Extract solution
    x = np.zeros(n)
    for i in range(n):
        col = tableau[:, i]
        if np.count_nonzero(col[:-1]) == 1 and np.count_nonzero(col) == 1:
            x[i] = tableau[np.where(col[:-1] == 1)[0][0], -1]

    return x

# 示例用法
if __name__ == "__main__":
    c = np.array([3, 2])
    A = np.array([[1, 2], [1, -1], [-1, 1]])
    b = np.array([4, 1, 2])
    solution = simplex(c, A, b)
    print("Solution:", solution)