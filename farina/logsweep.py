""" Functions that implement impulse response measurements using
A. Farina's logarithmic sweep method.
"""
import itertools as it
import numpy as np
from scipy import signal


def generate(f0, f1, fs, duration) -> np.ndarray:
    num_samples = int(duration * fs)
    f = np.logspace(np.log(f0), np.log(f1), num_samples, base=np.e)
    y = np.zeros(num_samples)
    curr_phase = 0

    # Using a brute force loop because signal.chirp produces errors with long
    # vectors.
    for n in range(len(y)):
        y[n] = np.sin(2 * np.pi * curr_phase)
        curr_phase += f[n] / fs
        if curr_phase > 1.0:
            curr_phase = curr_phase - 1.0

    return y


def apply_window(x: np.ndarray, ramp_length: int) -> None:
    win = signal.windows.hann(ramp_length * 2)
    x[:ramp_length] *= win[:ramp_length]
    x[-ramp_length:] *= win[-ramp_length:]


def inverse_filter(x: np.ndarray, f0: float, f1: float) -> np.ndarray:
    num_samples = x.shape[-1]
    start = 1.0
    end = 10 ** (-6 * np.log2(f1 / f0) / 20)
    envelope = np.logspace(np.log(start), np.log(end), num_samples, base=np.e)
    inv_filter = np.flip(x) * envelope
    return inv_filter


def impulse_response(sweep_wet: np.ndarray, inv_filter: np.ndarray):
    ir = signal.fftconvolve(sweep_wet, inv_filter)
    return ir


def thd_impulse_responses(
        ir: np.ndarray, f0: float, f1: float, num_harmonics: int, ramp_len: int = 256):
    length = ir.shape[-1] // 2
    deltas = [np.log(n) / np.log(f1 / f0) * length for n in range(2, num_harmonics + 2)]
    positions = [length] + [int(length - d) for d in deltas]
    thds = [ir[x0: x1] for x1, x0 in it.pairwise(positions)]

    win = signal.windows.hann(ramp_len * 2)

    for i in range(len(thds)):
        thds[i][-ramp_len:] *= win[-ramp_len:]

    return thds
