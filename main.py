import matplotlib.pyplot as plt
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

    return 0


if __name__ == "__main__":
    main()
