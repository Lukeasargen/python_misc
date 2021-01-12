import numpy as np

# Coefficient Matrix

# x, y

a = np.array(
    [
        [-1, 3],
        [3, -2]
    ]
)

# Dependent variables
b = np.array(
    [
        -7,
        -7
    ]
)

x = np.linalg.solve(a, b)

print("x :", x[0])
print("y :", x[1])
