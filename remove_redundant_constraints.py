import numpy as np

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
        if not np.all(np.abs(A_work[i]) < tol) or np.abs(b_work[i]) >= tol:
            nonzero_idx.append(i)

    # Extract the filtered constraints
    A_filtered = A[nonzero_idx, :]
    b_filtered = b[nonzero_idx]

    return A_filtered, b_filtered

# Example usage
if __name__ == "__main__":
    A = np.array([[1, 2, 3], [2, 4, 6], [1, 1, 1]], dtype=float)
    b = np.array([6, 12, 3], dtype=float)
    A_filtered, b_filtered = remove_redundant_constraints(A, b)
    print("Filtered A:\n", A_filtered)
    print("Filtered b:\n", b_filtered)