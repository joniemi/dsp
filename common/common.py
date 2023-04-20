import numpy as np


def to_dB(x: float) -> float:
    return 20 * np.log10(x)


def rms(x: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(x)))


def rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(rms(x))
