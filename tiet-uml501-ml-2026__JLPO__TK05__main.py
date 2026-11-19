import numpy as np
# Do not change the above code. Write your code below

# Read input
n, m = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(n)]

# Convert to NumPy array
arr = np.array(matrix)

# Compute sums
total_sum = np.sum(arr)
row_sums = np.sum(arr, axis=1)
col_sums = np.sum(arr, axis=0)

# Output
print(total_sum)
print(*row_sums)
print(*col_sums)