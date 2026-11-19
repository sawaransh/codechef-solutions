import numpy as np
# Do not change the code above
# Make sure to use equal, greater & less functions of numpy to solve the problem

n = int(input())

arr1 = np.array(list(map(int, input().split())))
arr2 = np.array(list(map(int, input().split())))

eq = np.equal(arr1, arr2)
gt = np.greater(arr1, arr2)
lt = np.less(arr1, arr2)

print(" ".join(map(str, eq)))
print(" ".join(map(str, gt)))
print(" ".join(map(str, lt)))