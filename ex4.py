import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def show(fig):
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Implementações
# ─────────────────────────────────────────────────────────────

def fft_radix2(x):
    N = len(x)
    if N == 1:
        return x.astype(complex)
    if N % 2 != 0:
        raise ValueError("Tamanho do vetor deve ser potência de 2.")
    even = fft_radix2(x[0::2])
    odd  = fft_radix2(x[1::2])
    W    = np.exp(-2j * np.pi * np.arange(N // 2) / N) * odd
    return np.concatenate([even + W, even - W])


def dft_direta(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return M @ x.astype(complex)


# ─────────────────────────────────────────────────────────────
# Parâmetros do sinal
# ─────────────────────────────────────────────────────────────
f0 = 50
spc = 2048
n_ciclos = 8
fs = f0 * spc
N = spc * n_ciclos

t = np.arange(N) / fs
x = np.sin(2 * np.pi * f0 * t)

print("=" * 60)
print("SEL0615 – Exercício 4")
print(f"  f0={f0} Hz | fs={fs} Hz | N={N} amostras")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# (a) Sinal + espectro
# ─────────────────────────────────────────────────────────────
print("\n(a) Plotando sinal + espectro...")

X = fft_radix2(x)

freqs = np.arange(N // 2) * fs / N
mag   = (2 / N) * np.abs(X[:N // 2])

fig = plt.figure(figsize=(14, 8))
gs  = gridspec.GridSpec(2, 1, hspace=0.45)

ax0 = fig.add_subplot(gs[0])
ax0.plot(t[:2*spc] * 1e3, x[:2*spc])
ax0.set_title("Sinal 50 Hz (2 ciclos)")
ax0.set_xlabel("Tempo (ms)")
ax0.grid()

ax1 = fig.add_subplot(gs[1])
mask = freqs <= 600
ax1.plot(freqs[mask], mag[mask])
ax1.axvline(50, linestyle="--", label="50 Hz")
ax1.set_title("Espectro")
ax1.legend()
ax1.grid()

show(fig)


# ─────────────────────────────────────────────────────────────
# (b) Custo computacional
# ─────────────────────────────────────────────────────────────
print("\n(b) Plotando custo computacional...")

pot = np.arange(4, 15)
tamanhos = 2 ** pot

custo_fft = (tamanhos / 2) * np.log2(tamanhos)
custo_dft = tamanhos ** 2

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, scale in zip(axes, ["linear", "log"]):
    ax.plot(tamanhos, custo_fft, label="FFT")
    ax.plot(tamanhos, custo_dft, label="DFT")
    ax.set_yscale(scale)
    ax.set_title(f"Escala {scale}")
    ax.legend()
    ax.grid()

show(fig)


# ─────────────────────────────────────────────────────────────
# (c) Tempos
# ─────────────────────────────────────────────────────────────
print("\n(c) Medindo tempos...")

N_c = 1024
x_c = x[:N_c]
RUNS = 30

t_fft, t_dft = [], []

for _ in range(RUNS):
    t0 = time.perf_counter()
    fft_radix2(x_c)
    t_fft.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    dft_direta(x_c)
    t_dft.append(time.perf_counter() - t0)

t_fft = np.array(t_fft) * 1e3
t_dft = np.array(t_dft) * 1e3

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(t_fft)
axes[0].set_title("FFT")

axes[1].hist(t_dft)
axes[1].set_title("DFT")

show(fig)


# ─────────────────────────────────────────────────────────────
# (d) Comparação
# ─────────────────────────────────────────────────────────────
mean_fft = t_fft.mean()
mean_dft = t_dft.mean()

print("\n(d) Comparação:")
print(f"Razão tempo (DFT/FFT): {mean_dft/mean_fft:.2f}x")


# ─────────────────────────────────────────────────────────────
# (e) THD
# ─────────────────────────────────────────────────────────────
print("\n(e) THD...")

harmonicos = np.array([
    [0.015, 0.002, 0.000, 0.012, 0.138],
    [0.220, 0.006, 0.170, 0.336, 0.051],
    [0.150, 0.003, 0.101, 0.016, 0.026],
    [0.000, 0.000, 0.000, 0.000, 0.016],
    [0.102, 0.062, 0.061, 0.087, 0.011],
    [0.084, 0.045, 0.044, 0.012, 0.008],
    [0.000, 0.000, 0.000, 0.000, 0.006],
])

sinais = ["6-pulse", "12-pulse", "SFC", "DC motor", "TCR"]

thd = np.sqrt(np.sum(harmonicos**2, axis=0)) * 100

fig, ax = plt.subplots()
ax.bar(sinais, thd)
ax.set_title("THD (%)")
ax.grid()

show(fig)