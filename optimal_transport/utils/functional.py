import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x - np.max(x)))
    return f_x