import numpy as np
# Do not change the above code. Write your code below

def fmt_line(vals):
    return " ".join(f"{round(float(v), 2):g}" if round(float(v), 2).is_integer() or str(round(float(v), 2)).endswith('0') else f"{float(v):.2f}" for v in vals)

# Input
n, m = map(int, input().split())
matrix = np.array([list(map(int, input().split())) for _ in range(n)])

# 1. Absolute value
abs_matrix = np.abs(matrix)

# 2. Percentiles
flat_pct = np.percentile(matrix, [25, 50, 75])
col_pct = np.percentile(matrix, [25, 50, 75], axis=0).T
row_pct = np.percentile(matrix, [25, 50, 75], axis=1).T

# 3. Statistics
flat_stats = [np.mean(matrix), np.median(matrix), np.std(matrix)]
col_mean = np.mean(matrix, axis=0)
col_med = np.median(matrix, axis=0)
col_std = np.std(matrix, axis=0)

row_mean = np.mean(matrix, axis=1)
row_med = np.median(matrix, axis=1)
row_std = np.std(matrix, axis=1)

# Output
# Absolute matrix
for row in abs_matrix:
    print(*row)

# Flatten percentiles
print(fmt_line(flat_pct))

# Column percentiles
for col in col_pct:
    print(fmt_line(col))

# Row percentiles
for row in row_pct:
    print(fmt_line(row))

# Flatten stats
print(fmt_line(flat_stats))

# Column stats
print(fmt_line(col_mean))
print(fmt_line(col_med))
print(fmt_line(col_std))

# Row stats
print(fmt_line(row_mean))
print(fmt_line(row_med))
print(fmt_line(row_std))