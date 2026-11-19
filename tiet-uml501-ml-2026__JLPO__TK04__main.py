import numpy as np
# Do not change the above code. Write your code below

# Take input for the size of the array
n = int(input())

# Take input for the n space-separated integers
arr = np.array(list(map(int, input().split())))

# Find unique values and their frequencies
values, counts = np.unique(arr, return_counts=True)

# np.unique is already sorted, so [0] grabs the smallest value in case of ties
most_frequent = values[np.argmax(counts)]

# Find all indices of this value
indices = np.where(arr == most_frequent)[0]

# Print results
print(most_frequent)
print(*indices)
