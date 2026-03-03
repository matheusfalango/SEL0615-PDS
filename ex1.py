import numpy as np
import matplotlib.pyplot as plt

# equivalente ao clear, close all, clc
plt.close('all')

N = 150000  # número de amostras muito elevado (quase contínuo)
P = 1       # número de períodos
A = 1       # amplitude do sinal

vet_amostras = np.arange(0, N*P)  # vetor de amostras
sinal = A * np.sin(2 * np.pi * vet_amostras / N)

plt.figure(1)
plt.plot(vet_amostras, sinal, 'r', linewidth=2)
plt.title('Senoide discreta')
plt.xlabel('Amostras')
plt.ylabel('Amplitude')
plt.grid(True)

plt.show()