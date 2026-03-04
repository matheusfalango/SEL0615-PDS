import numpy as np
import matplotlib.pyplot as plt # type: ignore

# equivalente ao clear, close all, clc
plt.close('all')

P = 1                                   # número de períodos
A = 1                                   # amplitude do sinal
F = int(input("Freq fundamental: "))    # frequência fundamental
FS = int(input("Freq amostragem: "))    # frequência de amostragem
N = int(FS/F)                           # número de amostras
TS = 1/FS                               # tempo de amostragem

if FS < 2*F:
    print("Aliasing!")

vet_amostras = np.arange(0, N*P)  # vetor de amostras
vet_tempo = vet_amostras*TS
sinal = A * np.sin(2 * np.pi * F * vet_tempo)

plt.figure(1)
plt.plot(vet_tempo, sinal, 'r', linewidth=2)    # sinal contínuo
plt.stem(vet_tempo, sinal)                      # sinal discreto
plt.title('Senoide discreta')
plt.xlabel('Tempo Discreto')
plt.ylabel('Amplitude')
plt.grid(True)

plt.show()