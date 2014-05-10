import numpy as np
#from scipy.optimize import minimize
from _minimize_joe import *


#the min of the function x**2 + 4*x + 16 should be at  x = -2, y = 12

def test_func(x):
    print("x: %f" %x)
    y = x**2 + 4*x + 16
    print("y: %f" %y)
    return y
    
x0 = 0

#the minimize function
# 1st arg: is the function that you would like to evaluate
# 2nd arg, a numpy array of variables that you would like to minimize over
res = minimize(test_func, x0, method='BFGS', options={'disp': True, 'maxiter': 5000})     
