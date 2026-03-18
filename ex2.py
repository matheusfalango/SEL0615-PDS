import math
import numpy as np
import matplotlib.pyplot as plt

# configuração de print (sem notação científica + 3 casas)
np.set_printoptions(precision=3, suppress=True)

# Função para Transformada Discreta de Fourier
def dft(x):

    N = len(x)
    X = [0] * N

    for m in range(N):
        soma = 0j

        for n in range(N):
            soma += x[n] * (math.cos(2*math.pi*n*m/N) - 1j * math.sin(2*math.pi*n*m/N))

        X[m] = soma

    return np.array(X)

# ler arquivos
def ler(nome):
    return np.loadtxt(nome, delimiter=";")

# Função Main
def main():

    # a) Ler arquivos
    vec_signal = ler("Sinal.txt")
    vec_time = ler("TimeStamp.txt")

    len_signal = len(vec_signal)
    len_time = len(vec_time)

    assert len_signal == len_time

    # b) Determinar frequência de amostragem
    sample_period_sum = 0

    for n in range(len_time - 1):
        sample_period_sum += vec_time[n+1] - vec_time[n]

    sample_period = sample_period_sum / (len_time - 1)
    sample_frequency = 1 / sample_period

    print(f"Frequência de amostragem: {sample_frequency:.3f}\n")

    # c) Determinar as frequências dos harmônicos presentes no sinal
    vec_freq_signal = dft(vec_signal)
    len_freq_signal = len(vec_freq_signal)

    # eixo de frequência
    freq_axis = np.arange(len_freq_signal) * sample_frequency / len_signal

    # ===== ALGORITMO MANUAL =====
    peaks_vec = []
    PEAK_THRESHOLD = 0.01

    for k in range(1, len_freq_signal // 2):

        mag_k = 2 * abs(vec_freq_signal[k]) / len_signal

        if mag_k > PEAK_THRESHOLD:
            peaks_vec.append(k)

    peaks_vec = np.array(peaks_vec)

    freqs = peaks_vec * sample_frequency / len_signal

    print("Harmônicos presentes no sinal (Hz):")
    print(freqs, "\n")

    # d) amplitudes
    mags = 2 * np.abs(vec_freq_signal[peaks_vec]) / len_signal

    print("Amplitudes dos harmônicos:")
    print(mags)

    # fundamental
    if len(freqs) > 0:
        idx_fund = np.argmin(freqs)

        freq_fund = freqs[idx_fund]
        amp_fund = mags[idx_fund]

        print(f"\nFrequência fundamental: {freq_fund:.3f}")
        print(f"Amplitude fundamental: {amp_fund:.3f}")


# Execução
if __name__ == "__main__":
    main()