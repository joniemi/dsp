""" Functions that implement impulse response measurements using
A. Farina's logarithmic sweep method.
"""
import numpy as np
from scipy import signal


def generate(f0, f1, fs, duration) -> np.ndarray:
    num_samples = int(duration * fs)
    t = np.linspace(0, duration, num_samples)
    y = signal.chirp(t, f0, duration, f1, method="logarithmic", phi=270)

    return y


def apply_window(x: np.ndarray, ramp_length):
    win = signal.windows.hann(ramp_length * 2)
    x[:ramp_length] *= win[:ramp_length]
    x[-ramp_length:] *= win[-ramp_length:]


def inverse_filter(f0, f1, num_samples):
    pass


def magnitude_response(x, inv_filter):
    # convolve with inverse filter
    # FFT and dB
    pass
