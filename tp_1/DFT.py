#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 10:41:53 2025

@author: ariel
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdsmodulos as pds

fs = 1000
N = fs

# DEFINICION DE FUNCIONES: 
#%% PRIMER FUNCION DFT: 
'''
def mi_funcion_DFT(xx):
    """
    Calcula la Transformada Discreta de Fourier (DFT) de una señal xx.
    
    Parameters
    ----------
    xx : array-like, tamaño N
        Señal real de entrada.

    Returns
    -------
    XX : array-like, tamaño N
        DFT de xx, valores complejos.
    """
    N = len(xx)
    XX = np.zeros(N, dtype=complex)  # Vector de salida
    
    for k in range(N):  # cada frecuencia
        suma = 0
        for n in range(N):  # suma sobre la señal
            suma += xx[n] * np.exp(-1j * 2 * np.pi * k * n / N)
        XX[k] = suma
    
    return XX
'''
#%% FUNCION DFT MEJORADA: 

def mi_funcion_DFT(xx):
    """
    Calcula la Transformada Discreta de Fourier (DFT) de una señal xx.
    usando producto matricial (sin for loops).
    
    Parameters
    ----------
    xx : array-like, tamaño N
        Señal real de entrada.

    Returns
    -------
    XX : array-like, tamaño N
        DFT de xx, valores complejos.
    """
    N = len(xx)
    n = np.arange(N) # vector fila (1 x N) -> tiempo
    k = np.expand_dims(n, axis=1) # vector columna (N x 1) -> frecuencia

    
    arg = ((-1) * 1j * 2 * np.pi) / N
    W = 1 * np.exp(arg * k * n)  # matriz NxN
    
    XX = np.dot(W, xx)  # producto matricial
    
    return XX  

#%% SENOIDAL:
def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=N , fs=fs):

    # grilla de sampleo temporal
    tt = np.arange(0, nn) / fs

    xx = vmax * np.sin(tt*2*np.pi*ff + ph) + dc

    return tt,xx

#%% PRUEBA DFT

tt, xx = mi_funcion_sen(vmax=1, dc=0, ff=5, ph=0, nn=N, fs=fs)
XX = mi_funcion_DFT(xx)

# Señal en el tiempo
plt.figure()
plt.plot(tt, xx)
plt.title("Señal en el tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

# Valor absoluto de la señal luego de la DFT en k bin
plt.figure()
plt.plot(np.abs(XX), "o")
plt.title("Transformada de Fourier propia - DFT")
plt.xlabel("k (bin de frecuencia)")
plt.ylabel("|X[k]|")
plt.grid(True)
plt.show()

# Valor absoluto de la señal luego de la DFT en frec
K = np.arange(N)
frec = K * fs / N

plt.figure()
plt.plot(frec, np.abs(XX))
plt.title('Espectro en función de la frecuencia (Hz)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|X[k]|')
plt.tight_layout()
plt.show()
#%% OTRAS SEÑALES DE PRUEBA:
    
def mi_funcion_trian(vmax=1, dc=0, ff=1, ph=0, nn=N, fs=fs):

    tt = np.arange(0, nn) / fs
    xx = dc + (2*vmax/np.pi) * np.arcsin(np.sin(2 * np.pi * ff * tt + ph))
    return tt, xx
    
def mi_funcion_cuad(vmax=1, dc=0, ff=1, ph=0, nn=N, fs=fs):

    tt = np.arange(0, nn) / fs
    xx = dc + vmax * np.sign(np.sin(2 * np.pi * ff * tt + ph))
    return tt, xx
    
tt_sq, sq = mi_funcion_cuad(vmax=1, dc=1, ff=10, ph=np.pi/4, nn=N, fs=fs)
tt_tria, tria = mi_funcion_trian(vmax=1, dc=1, ff=10, ph=np.pi/4, nn=N, fs=fs)

# Graficamos la señal en el tiempo
plt.figure()
plt.plot(tt_sq, sq)
plt.title("Señal cuadrada en el tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(tt_tria, tria)
plt.title("Señal triangular en el tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()


XX_sq = mi_funcion_DFT(sq)
XX_tria = mi_funcion_DFT(tria)

#GRAFICAMOS 
plt.figure()
plt.plot(np.abs(XX_sq), "o")
plt.title("Transformada de Fourier propia: cuadrada- DFT")
plt.xlabel("k (bin de frecuencia)")
plt.ylabel("|X[k]|")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(frec, np.abs(XX_sq))
plt.title('Espectro en función de la frecuencia (Hz)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud')
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(np.abs(XX_tria), "o")
plt.title("Transformada de Fourier propia: triangular - DFT")
plt.xlabel("k (bin de frecuencia)")
plt.ylabel("|X[k]|")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(frec, np.abs(XX_tria))
plt.title('Espectro en función de la frecuencia (Hz)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud')
plt.tight_layout()
plt.show()
#%% BONUS: 
    
# Ruido uniforme con varianza = 4
# Queremos varianza sigma^2 = 4 => sigma = 2
# Ruido uniforme en [-a, a], su varianza es a^2 / 3
# Entonces a = sqrt(3*sigma^2) = sqrt(12)
a = -np.sqrt(12*4)/2
b =  np.sqrt(12*4)/2
ruido = np.random.uniform(a, b, N)

# Señal total
xx_ruido = xx + ruido

XX_ruido = mi_funcion_DFT(xx_ruido)
XX_fft_ruido = np.fft.fft(xx_ruido)

#%% GRAFICOS

# Señales en el tiempo
plt.figure()
plt.plot(tt, xx, label="Senoidal pura")
plt.plot(tt, xx_ruido, label="Seno + ruido", alpha=0.7)
plt.title("Señal en el tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.show()

# Espectros
plt.figure()
plt.plot(np.abs(XX_ruido), "x", label="DFT propia")
plt.plot(np.abs(XX_fft_ruido), "o", label="FFT NumPy")
plt.title("Espectro de la señal con ruido")
plt.xlabel("k (bin de frecuencia)")
plt.ylabel("|X[k]|")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(frec, np.abs(XX_ruido))
plt.title('Espectro en función de la frecuencia (Hz)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|X[k]|')
plt.tight_layout()
plt.show()

#% EXTRA VARIANZA VARIAS:
varianzas = [0.1, 1, 4, 10]

plt.figure(figsize=(12, 8))

for i, var in enumerate(varianzas):
    # Calculo de límites para ruido uniforme con varianza deseada
    a = -np.sqrt(12 * var) / 2
    b =  np.sqrt(12 * var) / 2
    ruido_variable = np.random.uniform(a, b, N)
    
    # Señal con ruido
    xx_ruidosa = xx + ruido_variable
    
    # Cálculo de la DFT
    espectro = np.abs(mi_funcion_DFT(xx_ruidosa))
    
    # Gráfico
    plt.subplot(len(varianzas), 1, i+1)
    plt.plot(frec, espectro)
    plt.title(f'Varianza del ruido: {var}')
    plt.ylabel('Magnitud')

plt.xlabel('Frecuencia [Hz]')
plt.tight_layout()
plt.show()