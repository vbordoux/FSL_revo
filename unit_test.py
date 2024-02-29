import matplotlib.pyplot as plt
from scipy.signal import freqz, butter, lfilter
from torch import from_numpy
import numpy as np

def test_filter_response_stability(audio, order, cutoffs, sr):
    np_waveform = butter_bandpass_filter(audio, cutoffs=cutoffs, fs=sr, order=order)
    
    b, a = butter_bandpass(cutoffs=cutoffs, fs=sr, order=order)
    w,h = freqz(b, a, fs=sr, worN= 8*cutoffs[1])

    plt.subplot(2,1,1)
    plt.semilogx(w,np.abs(h), 'r' )
    plt.axvline(cutoffs[0], color='k')
    plt.axvline(cutoffs[1], color='b')
    plt.xlim(0, 4*cutoffs[-1])
    plt.title(' Bandpass filter tryout')
    plt.xlabel(' F(Hz)')
    plt.grid()

    T = 5.0
    n = int(T*sr)
    t = np.linspace(0, T, n, endpoint=False)
    plt.subplot(2,1,2)
    plt.plot(t, audio[0][:n], 'b-', linewidth=2, label='data')
    plt.plot(t, np_waveform[0][:n], 'g-', label='filtered data')
    plt.grid()
    plt.legend()
    plt.subplots_adjust(hspace=0.35)

    plt.show()


def butter_bandpass(cutoffs, fs, order=5):
    return butter(order, cutoffs, fs=fs, btype='band', analog=False)

def butter_bandpass_filter(audio, cutoffs, fs, order=5):
    b, a = butter_bandpass(cutoffs, fs, order=order)
    y = lfilter(b, a, audio)
    return y

