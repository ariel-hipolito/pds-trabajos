#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulación DSP - Estimación de amplitud y frecuencia con ruido
y diferentes ventanas
@author: ariel
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window

#%% VARIABLES GLOBALES:

N = 1000   # cantidad de muestras
M = 200    # cantidad de realizaciones 
fs = N     # frecuencia de muestreo (Hz)
omega0 = np.pi/2       # frecuencia central
a0 = np.sqrt(2)        # amplitud -> potencia = 1 W
SNR_dbs = [3, 10]      # SNRs a simular

ventanas = {
    "rectangular": np.ones(N),
    "flattop": get_window("flattop", N),
    "blackmanharris": get_window("blackmanharris", N),
    "hann": get_window("hann", N)
}

tt = np.arange(0, N)   # vector de tiempo discreto

N_fft = 8*N  # Zero-padding para mejor resolución de frecuencia (bonus)
#%% PRESCRIBIR SNR:
    
# X[n] = a0 * sin(omega1 * n) + na(n)

# omega1 = omega0 + fr * (2*pi/N)

# na ~ N(0, var**2) 

# LOOP PRINCIPAL
for snr_db in SNR_dbs:
    
    print(f"\n### Simulación con SNR = {snr_db} dB ###")
    
    # Varianza del ruido para el SNR prescrito
    sigma2 = a0**2 / (2*(10**(snr_db/10)))
    
    # Frecuencias aleatorias alrededor de omega0
    frs = np.random.uniform(-2, 2, M)
    omega1 = omega0 + frs * 2 * np.pi / N

    # Expandir dimensiones
    tt_matricial = np.tile(tt.reshape(N, 1), (1, M))
    omega1_matricial = np.tile(omega1.reshape(1, M), (N, 1))
    
    # Señal senoidal + ruido
    sin = a0 * np.sin(omega1_matricial * tt_matricial)
    ruido = np.random.normal(0, np.sqrt(sigma2), (N, M))
    sin_ruidosa = sin + ruido
    
    # Frecuencias de la FFT
    freqs = 2*np.pi * np.arange(N_fft) / N_fft
    
    for nombre, w in ventanas.items():
        
        print(f"  Ventana: {nombre}")
        
        # Aplicar ventana
        w_matricial = np.tile(w.reshape(N,1), (1, M))
        sin_w = sin_ruidosa * w_matricial
        
        # FFT
        Xw = np.fft.fft(sin_w, N_fft, axis=0)
        #HASTA ACA ESTA TODO OK!
        
        # Estimadores
        idx0 = np.argmin(np.abs(freqs - omega0))
        Gw = np.sum(w) / N   # Ganancia de la ventana
        estim_amplitud = (2 / (N * Gw)) * np.abs(Xw[idx0, :])
        
        idx_max = np.argmax(np.abs(Xw), axis=0)
        estim_frec = freqs[idx_max]
        
        # Estadísticos
        mu_a = np.mean(estim_amplitud)
        sesgo_a = mu_a - a0
        var_a = np.var(estim_amplitud)
        
        mu_f = np.mean(estim_frec)
        mu_f_teo = np.mean(omega1)
        sesgo_f = mu_f - mu_f_teo
        var_f = np.var(estim_frec)
        
        print(f"    Amplitud: sesgo={sesgo_a:.4f}, var={var_a:.4f}")
        print(f"    Frecuencia: sesgo={sesgo_f:.4e}, var={var_f:.4e}")
        
        # ---- HISTOGRAMAS ----
        
        # Amplitud en dB
        estim_amp_db = 20*np.log10(estim_amplitud)
        
        plt.figure(figsize=(10,4))
        plt.hist(estim_amp_db, bins=30, color="skyblue", edgecolor="k")
        plt.title(f"Histograma de amplitud estimada (dB)\nVentana: {nombre}, SNR={snr_db} dB")
        plt.xlabel("Amplitud estimada [dB]")
        plt.ylabel("Frecuencia de ocurrencia")
        plt.grid(True)
        plt.show()
        
        # Frecuencia en dB
        estim_freq_db = 20*np.log10(np.abs(estim_frec) + 1e-12)  # evitar log(0)
        
        plt.figure(figsize=(10,4))
        plt.hist(estim_freq_db, bins=30, color="salmon", edgecolor="k")
        plt.title(f"Histograma de frecuencia estimada (dB)\nVentana: {nombre}, SNR={snr_db} dB")
        plt.xlabel("Frecuencia estimada [dB]")
        plt.ylabel("Frecuencia de ocurrencia")
        plt.grid(True)
        plt.show()

'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 09:11:39 2025

@author: ariel
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdsmodulos as pds
from scipy.signal import get_window

#%% VARIABLES GLOBALES:
    
N = 1000   # cantidad de muestras
M = 200    # cantidad de realizaciones 
fs = N     # frecuencia de muestreo (Hz)
omega0 = np.pi/2       # frecuencia central
a0 = np.sqrt(2)        # amplitud -> potencia = 1 W
SNR_dbs = [3, 10]

ventanas = {
    "rectangular": np.ones(N),
    "flattop": get_window("flattop", N),
    "blackmanharris": get_window("blackmanharris", N),
    "hann": get_window("hann", N)
}


"""
def prescribir_SNR(N, omega0, fr, a0, sigma2):
    
    n = np.arrange(N)
    rad_fr = fr*np.pi/180
    omega1 = omega0 + rad_fr

    return 
"""

tt = np.arange(0, N)

snr_db = 3
sigma2 = a0**2 / (2*(10**(snr_db/10)))

frs = np.random.uniform(-2, 2, M)
omega1 = omega0 + frs * 2 * np.pi / N

# MATRIZ: ESTO ESTA BIEN
tt_matricial = np.tile(tt.reshape(N, 1), (1, M))
omega1_matricial = np.tile(omega1.reshape(1, M), (N, 1))

# Señal senoidal + ruido
sin = a0 * np.sin(omega1_matricial * tt_matricial)
ruido = np.random.normal(0, np.sqrt(sigma2), (N, M))
sin_ruidosa = sin + ruido

# Frecuencias de la FFT
freqs = 2*np.pi * np.arange(N) / N


w = get_window("hann", N)
w_matricial = np.tile(w.reshape(N,1), (1, M))

sin_w = sin_ruidosa * w_matricial

Xw = np.fft.fft(sin_w, N, axis=0)


#HASTA ACA ESTA TODO OK!
freqs = 2*np.pi * np.arange(Xw.shape[0]) / Xw.shape[0]

idx0 = np.argmin(np.abs(freqs - omega0))
estim_amplitud = np.abs(Xw[idx0, :])

idx_max = np.argmax(np.abs(Xw), axis=0)    
estim_frec = freqs[idx_max]


mu_a = np.mean(estim_amplitud)
sesgo_a = mu_a - a0
var_a = np.var(estim_amplitud)

mu_f = np.mean(estim_frec)
mu_f_teo = np.mean(omega1)
sesgo_f = mu_f - mu_f_teo
var_f = np.var(estim_frec)

k = 0
plt.figure(figsize=(10,5))
plt.plot(freqs, 20*np.log10(np.abs(Xw[:,k])))  # dB, evito log(0)
plt.title("Espectro de la realización 0")
plt.xlabel("Frecuencia [rad/muestra]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)
plt.show()

'''
'''
pot_signal = np.mean(sin**2)     #  P = (1/N) Σ |x[n]|² 
pot_noise = 10**(- (snr_db/10) )


noise = np.random.normal(0, np.sqrt(pot_noise) ,len(sin))
signal_noisy = sin + noise

verificacion = 10 * np.log10(np.var(sin)/np.var(noise))


#%% ESPECTRO: 
fft = np.fft.fft(signal_noisy)/N
fft_mag_pos = 20 * np.log10(np.abs(fft[:(N//2) + 1]))
fft_mag_pos[1:-1] *= 2  # duplicar energía de los componentes positivos (excepto extremos)
bins_pos = np.arange(0, N//2 + 1)

plt.figure(figsize=(10,5))
plt.plot(bins_pos, fft_mag_pos, "x")
plt.xlabel("k (bin de frecuencia)")
plt.ylabel("|X[k]| db")
plt.legend()
plt.grid(True)
plt.show()
'''