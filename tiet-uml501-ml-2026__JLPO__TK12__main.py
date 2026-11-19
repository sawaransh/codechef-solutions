import numpy as np
# Do not change the above code. Write your code below

# Input
arr = np.array(list(map(float, input().split())))

# Integer elements
int_elements = arr[arr == arr.astype(int)]

# Float (non-integer) elements
float_elements = arr[arr != arr.astype(int)]

# Output
print(*int_elements)
print(*float_elements)