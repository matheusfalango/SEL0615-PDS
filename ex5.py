import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import wavfile
from scipy.fft import fft, ifft
import warnings

warnings.filterwarnings('ignore')

# --- PARTE 1: PROCESSAMENTO DE IMAGEM ---
# Carregamento e separação dos canais RGB
img = np.array(Image.open('imagem.jpg'))
R, G, B = img[:,:,0], img[:,:,1], img[:,:,2] [cite: 356, 399]

def criar_mascara_passa_baixa(shape, raio):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((Y - crow)**2 + (X - ccol)**2)
    return (dist <= raio).astype(float) [cite: 364, 400]

def aplicar_mascara_canal(canal, mascara):
    F = np.fft.fft2(canal)
    Fshift = np.fft.fftshift(F)
    Fshift *= mascara
    F_inv = np.fft.ifftshift(Fshift)
    resultado = np.fft.ifft2(F_inv)
    return np.clip(np.abs(resultado), 0, 255).astype(np.uint8) [cite: 373, 401]

def reconstruir_rgb(R, G, B, mascara):
    canais_filtrados = [aplicar_mascara_canal(c, mascara) for c in (R, G, B)]
    return np.stack(canais_filtrados, axis=2) [cite: 373, 403]

# Geração das máscaras e filtragem
mascara1 = criar_mascara_passa_baixa(R.shape, raio=30)
mascara2 = criar_mascara_passa_baixa(R.shape, raio=80)

img_filt1 = reconstruir_rgb(R, G, B, mascara1) # Filtro mais restritivo
img_filt2 = reconstruir_rgb(R, G, B, mascara2) # Filtro mais permissivo [cite: 373, 403]

# --- PARTE 2: PROCESSAMENTO DE ÁUDIO ---
sr, audio = wavfile.read('musica.wav')
audio = audio.astype(np.float64) [cite: 404]

def comprimir_canal(canal, percentual):
    F = fft(canal)
    # Mantém apenas os top-p% coeficientes de maior magnitude
    limiar = np.percentile(np.abs(F), 100 - percentual)
    F_comp = np.where(np.abs(F) >= limiar, F, 0)
    return np.real(ifft(F_comp)) [cite: 385, 406]

def comprimir_audio(audio, pct):
    if audio.ndim == 1:
        return comprimir_canal(audio, pct)
    # Trata áudio estéreo (2 canais)
    return np.stack([comprimir_canal(audio[:, i], pct) for i in range(audio.shape[1])], axis=1) [cite: 408]

audio_5pct = comprimir_audio(audio, 5)
audio_1pct = comprimir_audio(audio, 1) [cite: 408]