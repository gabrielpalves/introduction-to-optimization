import numpy as np

a = np.array([1, 2])
b = np.array([
    [3, 4],
    [5, 2],
    [1, 1]
    ])

c = b.T
print(np.matmul(a, c))