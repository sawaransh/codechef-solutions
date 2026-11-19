import numpy as np
# Do not change the above code. Write your code below

# Read input
n, m = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(n)]

# Convert to NumPy array
arr = np.array(matrix)

# Flatten the array
flattened_arr = arr.flatten()

# Print result
print(*flattened_arr)