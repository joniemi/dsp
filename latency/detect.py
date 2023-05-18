import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, correlation_lags


def detect_latency(x: np.ndarray, y: np.ndarray) -> int:
    """ Detects the latency of signal `y` with respect to signal `x`
    using cross correlation. Works best with noise signals.
    """
    c = correlate(y, x, mode="full")
    lags = correlation_lags(x.size, y.size, mode="full")
    latency = lags[np.argmax(c)]
    return latency


def plot_xcorr(x: np.ndarray, y: np.ndarray) -> None:
    c = correlate(y, x, mode="full")
    n = len(c)
    x_axis = np.linspace(-0.5 * n, 0.5 * n, n)

    plt.figure()
    plt.grid()
    plt.plot(x_axis, c)
    plt.show()
