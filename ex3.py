import numpy as np
import matplotlib.pyplot as plt

# Function Main
def main():
    
    # Parameters
    A = 1
    F = 50
    N = 128
    FS = F * N
    TS = 0.11

    if FS < 2 * F:
        print("Aliasing!")

    vet_tempo = np.arange(0, TS, 1/FS)

    signal = A * np.sin(2 * np.pi * F * vet_tempo)
    signal1 = 0.5 * np.sin(2 * np.pi * 300 * vet_tempo)
    signal2 = 0.25 * np.sin(2 * np.pi * 500 * vet_tempo)

    # sum of signals
    signal = signal + signal1 + signal2
    len_signal = len(signal)

    # FFT
    freq_signal = np.fft.fft(signal)

    # Freqs
    freqs = np.fft.fftfreq(len_signal, 1/FS)

    half = len_signal // 2

    freqs_plot = freqs[:half+1]
    mag = np.abs(freq_signal) / len_signal
    mag = mag[:half+1]
    mag[1:-1] = 2 * mag[1:-1]

    # ---
    # item a
    # ---
    plt.figure()
    plt.stem(freqs_plot, mag)
    plt.title("Spectrum (DFT off window)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 1000)
    plt.grid()
    plt.show()

    # ---
    # item b
    # ---
    energy = 0
    for i in range(len_signal):
        f = i * FS / len_signal
        if (0 <= f <= 1000) or (FS-1000 <= f <= FS):
            energy += np.abs(freq_signal[i])**2

    energy = energy / len_signal
    print("Energy (off window):", energy)

    # ---
    # item c
    # ---
    windows = {
        "Retangular": np.ones(len_signal),
        "Hann": np.hanning(len_signal),
        "Hamming": np.hamming(len_signal),
        "Blackman": np.blackman(len_signal)
    }

    pow_windows = {}

    for type, w in windows.items():

        G = np.mean(w)

        wsignal = signal * w
        freq_wsignal = np.fft.fft(wsignal)

        # normalization with gain
        wmag = np.abs(freq_wsignal) / (len_signal * G)
        wmag = wmag[:half+1]
        wmag[1:-1] = 2 * wmag[1:-1]

        # Plot
        plt.figure()
        plt.stem(freqs_plot, wmag)
        plt.title(f"Spectrum with Window {type}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.xlim(0, 1000)
        plt.grid()
        plt.show()

        # ---
        # item d
        # ---
        wenergy = 0
        for i in range(len_signal):
            f = i * FS / len_signal
            if (0 <= f <= 1000) or (FS-1000 <= f <= FS):
                wenergy += np.abs(freq_wsignal[i])**2

        wenergy = wenergy / len_signal
        pow_windows[type] = wenergy

    # Print
    print("\nEnergy with Windows:")
    for type, val in pow_windows.items():
        print(f"{type}: {val}")


# Execute
if __name__ == '__main__': 
    main()