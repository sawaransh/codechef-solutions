import numpy as np
# Do not change the above code. Write your code below

# Input
n = int(input())
arr = np.array(list(map(int, input().split())))

# 1. Sorted array
sorted_arr = np.sort(arr)

# 2. Indices of sorted array
sorted_indices = np.argsort(arr)

# 3. 4 smallest elements
smallest_4 = sorted_arr[:4]

# 4. 5 largest elements
largest_5 = sorted_arr[-5:]

# Output
print(*sorted_arr)
print(*sorted_indices)
print(*smallest_4)
print(*largest_5)