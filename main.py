import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

from common import plots, common
from farina import logsweep


def main():
    args = parse_args()
    func = DEMOS[args.demo]
    func()

    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "demo", choices=DEMOS.keys(), type=str,
        help="Choose which package you would like to run a demonstration of."
    )
    args = parser.parse_args()
    return args


def calculate_magnitude(x: np.ndarray, nfft, scale: str = "db") -> np.ndarray:
    magn = np.abs(fft(x, n=nfft)) / (nfft / 2)

    if scale == "db":
        magn = common.to_dB(magn)

    return magn


def farina_demo():
    fs = 48_000
    f0, f1 = 20, 20_000
    duration = 2.0
    x = logsweep.generate(f0, f1, fs, duration)
    logsweep.apply_window(x, 2048)

    N = x.shape[-1]
    nfft = N
    freqs = fftfreq(nfft, d=1 / fs)

    plt.figure()
    plt.plot(x)

    plt.figure()
    plt.semilogx(freqs, calculate_magnitude(x, nfft))
    plt.title('Magnitude response')
    plt.xlim([10, fs // 2])
    plt.grid(True)

    # simulate distortion
    y = x * 1.00
    y = np.clip(y, -1.0, 1.0)

    inv_filter = logsweep.inverse_filter(x, f0, f1)
    ir = logsweep.impulse_response(y, inv_filter)

    plt.figure()
    plt.semilogx(freqs, calculate_magnitude(x, nfft), label="sweep")
    plt.semilogx(freqs, calculate_magnitude(inv_filter, nfft), label="inverse filter")
    plt.semilogx(freqs, calculate_magnitude(ir[-N:], nfft), label="sweep filtered")
    plt.title('Magnitude response')
    plt.legend()
    plt.xlim([10, fs // 2])
    plt.grid(True)

    thd_responses = logsweep.thd_impulse_responses(ir, f0, f1, 5)

    plt.figure()
    nfft = 2 ** 15
    freqs = fftfreq(nfft, 1 / fs)
    plt.semilogx(
        freqs,
        calculate_magnitude(ir[-N:], nfft),
        label="freq. resp"
    )

    for i, thd_resp in enumerate(thd_responses):
        plt.semilogx(freqs, calculate_magnitude(thd_resp, nfft), label=f"harmonic #{i}")

    plt.legend()
    plt.grid()
    plt.xlim([10, fs // 2])

    plt.figure()
    plt.plot(ir)

    plt.show()


def latency_demo():
    return 0


DEMOS = {
    "farina": farina_demo,
    "latency": latency_demo,
}


if __name__ == "__main__":
    main()
