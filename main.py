import matplotlib.pyplot as plt
import numpy as np

from common import plots
from farina import logsweep


def main():
    fs = 48_000
    f0, f1 = 10, fs / 2
    duration = 1.0
    x = logsweep.generate(f0, f1, fs, duration)
    logsweep.apply_window(x, 2048)

    plt.figure()
    plt.plot(x)
    plots.plot_magnitude_and_phase(x, fs)

    # simulate distortion
    y = x * 1.05
    y = np.clip(y, -1.0, 1.0)

    inv_filter = logsweep.inverse_filter(x, f0, f1)
    ir = logsweep.impulse_response(y, inv_filter)
    thd_responses = logsweep.thd_impulse_responses(ir, f0, f1, 4)

    plt.figure()

    for thd_resp in thd_responses:
        plt.semilogx(thd_resp)

    plt.show()

    return 0


if __name__ == "__main__":
    main()
