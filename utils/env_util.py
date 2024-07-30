import random
import cv2
import numpy as np


def adjust_action(t, threshold):
    # Explore as much as possible in the early stage

    rn = random.randint(0, 9)
    if t < threshold:
        if rn in [0, 1, 2, 3, 4]:
            return np.array([-0.1, 1, 0])
        else:
            return np.array([0.1, 1, 0])
    if rn in [0, 9]:
        return np.array([0, 0, 0])
    elif rn in [1, 2]:
        return np.array([0, random.random(), 0])
    elif rn in [3, 4]:
        return np.array([-random.random(), 0, 0])
    elif rn in [5, 6]:
        return np.array([random.random(), 0, 0])
    elif rn in [7, 8]:
        return np.array([0, 0, random.random()])

def adjust_obs(obs, size):
    obs = cv2.resize(obs, dsize=size, interpolation=cv2.INTER_CUBIC)
    return obs.astype('float32') / 255.