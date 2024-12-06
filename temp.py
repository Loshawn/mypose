import numpy as np

a = np.arange(133*3)

b = a[:17*3]
c = a[17*3:23*3].reshape(-1, 3)/3
print(a, b,c)