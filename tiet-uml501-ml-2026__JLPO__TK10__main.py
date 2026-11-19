import numpy as np
# Do not change the above code. Write your code below

# Input
n = int(input())
arr = np.array(list(map(float, input().split())))

# Operations
floor_vals = np.floor(arr).astype(int)
ceil_vals = np.ceil(arr).astype(int)
trunc_vals = np.trunc(arr).astype(int)
round_vals = np.round(arr).astype(int)

# Output
print(*floor_vals)
print(*ceil_vals)
print(*trunc_vals)
print(*round_vals)