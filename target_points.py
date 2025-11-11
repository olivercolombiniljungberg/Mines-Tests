import numpy as np

def target_point(init_pos, t):
    target = init_pos + 0.1 * t * np.array([1.0,1.0])
    return np.array(target)