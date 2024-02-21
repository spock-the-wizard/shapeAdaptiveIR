import numpy as np


if __name__ == "__main__":
    mat = np.array([[1,2,3,],
                    [4,5,6,],
                    [7,8,9,]])
    vec = np.array([1,2,3])
    # [...,None]

    res = mat@vec
    print("[Test 1] Matrix Vector Multiplication")
    print(res)