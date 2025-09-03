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

#%% VARIABLES GLOBALES:
    
N = 1000   # cantidad de muestras
fs = N # frecuencia de muestreo (Hz)
df = fs/N

#%% SENOIDAL: 

def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=N , fs=fs):

    # grilla de sampleo temporal
    tt = np.arange(0, nn) / fs

    xx = vmax * np.sin(tt*2*np.pi*ff + ph) + dc

    return tt,xx

#%% FFT: 

time_sin_1, sin_1= mi_funcion_sen(vmax=1, dc=0, ff=(fs/4), ph=0, nn=N, fs=fs)
sin_fft_1 = np.fft.fft(sin_1)

time_sin_2, sin_2= mi_funcion_sen(vmax=1, dc=0, ff=((fs/4) + df/2), ph=0, nn=N, fs=fs)
sin_fft_2 = np.fft.fft(sin_2)

#%% ESPECTRO: representacion en logaritmo - chusmear pag 885 holton - lobulos 
#secundario masbajos mejor 

# 

db1 = 20 * np.log10(np.abs(sin_fft_1))
plt.figure()
plt.plot(db1, "x")
#plt.title("Espectro de la señal con ruido")
plt.xlabel("k (bin de frecuencia)")
plt.ylabel("|X[k]| db")
plt.legend()
plt.grid(True)
plt.show()

db2 = 20 * np.log10(np.abs(sin_fft_2))
plt.figure()
plt.plot(db2, "x")
#plt.title("Espectro de la señal con ruido")
plt.xlabel("k (bin de frecuencia)")
plt.ylabel("|X[k]| db")
plt.legend()
plt.grid(True)
plt.show()