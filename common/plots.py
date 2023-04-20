import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from common.common import to_dB


def plot_freqz(w, h):
    plt.subplot(211)
    plt.plot(w, to_dB(h), 'b')
    plt.xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    plt.grid(True)
    plt.subplot(212)
    plt.plot(w, np.imag(h), 'r')
    plt.xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    plt.grid(True)


def plot_magnitude_and_phase(y, fs, label=''):
    Y = fft(y, fs)
    plt.figure()
    plt.subplot(211)
    plt.semilogx(to_dB(np.abs(Y) / (fs / 2)), label=label)
    plt.title('Magnitude response')
    #plt.ylim([-30, 2])x
    plt.xlim([fs/1000, fs/2])
    plt.grid(True)
    plt.subplot(212)
    plt.semilogx(np.imag(Y), 'r')
    plt.xlim([fs/1000, fs/2])
    plt.title('Phase response')
    plt.grid(True)
    plt.show()
