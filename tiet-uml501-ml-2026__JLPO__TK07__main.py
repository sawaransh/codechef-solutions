import numpy as np
# Do not change the above code. Write your code below

# Input first matrix
n, m = map(int, input().split())
mat1 = np.array([list(map(int, input().split())) for _ in range(n)])

# Input second matrix
p, q = map(int, input().split())
mat2 = np.array([list(map(int, input().split())) for _ in range(p)])

# Matrix multiplication
prod = np.matmul(mat1, mat2)

# Covariance (combine both matrices row-wise)
combined = np.vstack((mat1, mat2))
cov_matrix = np.cov(combined)

# Output multiplication result
for row in prod:
    print(*row)

# Output covariance matrix
for row in cov_matrix:
    print(" ".join(f"{val:.2f}" for val in row))