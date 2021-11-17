import numpy as np

def add(x):
    x[0] = 100


x = np.array([1,2,3,4])
y = np.array([1,10,100,1000])

z = ','.join(y)
print(z)