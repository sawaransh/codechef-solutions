import numpy as np
# Do not change the above code. Write your code below

# Read input
n = int(input())
elements = list(map(int, input().split()))

# Convert to NumPy array
arr = np.array(elements)

# Reverse the array
reversed_arr = arr[::-1]

# Print result
print(*reversed_arr)