import numpy as np
# Do not change the above code. Write your code below

# Read input
n, m = map(int, input().split())
mat1 = [list(map(int, input().split())) for _ in range(n)]

p, q = map(int, input().split())
mat2 = [list(map(int, input().split())) for _ in range(p)]

# Convert to numpy arrays
arr1 = np.array(mat1)
arr2 = np.array(mat2)

# Flatten
flat1 = arr1.flatten()
flat2 = arr2.flatten()

# Inner product
inner_prod = np.inner(flat1, flat2)

# Outer product
outer_prod = np.outer(flat1, flat2)

# Cartesian product
cartesian_prod = np.array(np.meshgrid(flat1, flat2)).T.reshape(-1, 2)

# Output
print(inner_prod)
for row in outer_prod:
    print(*row)
for pair in cartesian_prod:
    print(f"({pair[0]}, {pair[1]})")