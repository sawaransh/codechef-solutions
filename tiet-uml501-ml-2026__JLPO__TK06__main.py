import numpy as np
# Do not change the above code. Write your code below

def fmt(val):
    if isinstance(val, complex):
        val = val.real
    if abs(val) < 1e-9:
        val = 0.0
    return f"{val:.2f}"

# Input
n = int(input())
matrix_list = [list(map(int, input().split())) for _ in range(n)]

# Must use NumPy array
matrix = np.array(matrix_list)

# 1. Sum of diagonal elements
diag_sum = np.trace(matrix)
print(diag_sum)

# 2 & 3. Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)

# Print eigenvalues
print(" ".join(fmt(val) for val in eigenvalues))

# Print eigenvectors (each column is an eigenvector)
for i in range(n):
    vec = eigenvectors[:, i]
    # Standardize sign (optional but helps consistency)
    if vec[0].real < 0:
        vec = -vec
    print(" ".join(fmt(val) for val in vec))

# 4. Inverse (must always attempt using NumPy)
try:
    inv_matrix = np.linalg.inv(matrix)
    for row in inv_matrix:
        print(" ".join(fmt(val) for val in row))
except np.linalg.LinAlgError:
    print("Not Invertible")

# 5. Determinant
det = np.linalg.det(matrix)
print(fmt(det))