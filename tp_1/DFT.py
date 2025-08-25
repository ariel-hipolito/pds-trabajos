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
#%% 

# Datos generales de la simulación
N = 10   # cantidad de muestras
fs = N # frecuencia de muestreo (Hz)
df = fs/N # resolución espectral

ts = 1/fs # tiempo de muestreo
fo = 5 #Hz
    
# grilla de sampleo temporal
tt = np.arange(stop=1, step=ts)

# senoidal a trabajar 
sig_type = np.sin(tt*2*np.pi*fo)

#%%

#def my_funtion_DFT( sig_type ):
    
sig_DFT = np.zeros(N, dtype = np.complex128())
    
for  k in range(0, N-1):
        
    for n in range(0, N-1):
            
        sig_DFT[k] += sig_type[n] * 1*np.exp(((-1) * 1j *2*np.pi*n / N) *k)  
            
    #return sig_DFT

# Invocamos a nuestro testbench exclusivamente: 
#sig_DFT = my_funtion_DFT( sig_type)

#plt.plot(tt, sig_DFT )

"""#%% Presentación gráfica de los resultados
    
plt.figure(1)
line_hdls = plt.plot(tt, sig_DFT)
plt.title('Señal: DFT' )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
    
plt.show()

"""
"""arg = (-1) * 1j *2*np.pi./ N) * k
b[n] = 1*np.exp(arg*n)

sig_DFT = np.dot(sig_type,b) // np.dot es el producto escalar de dos arrays o matrices 


"""