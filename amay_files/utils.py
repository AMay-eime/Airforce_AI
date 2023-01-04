import numpy as np
from math import sqrt

def norm_vec(vec:np.ndarray):
    magni = sqrt(pow(vec[0], 2) + pow(vec[1], 2) + pow(vec[2], 2))
    return vec / magni

def magnitude(vec:np.ndarray):
    magni = sqrt(pow(vec[0], 2) + pow(vec[1], 2) + pow(vec[2], 2))
    return magni
    