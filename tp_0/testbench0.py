#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Spyder(Phyton 3.10)

@author: Ariel Hipolito

Descripción:
------------
En este primer trabajo comenzaremos por diseñar un generador de señales que 
utilizaremos en las primeras simulaciones que hagamos. La primer tarea 
consistirá en programar una función que genere señales senoidales y que permita 
parametrizar:

 - la amplitud máxima de la senoidal (volts)
 - su valor medio (volts)
 - la frecuencia (Hz)
 - la fase (radianes)
 - la cantidad de muestras digitalizada por el ADC (# muestras)
 - la frecuencia de muestreo del ADC.

es decir que la función que uds armen debería admitir se llamada de la 
siguiente manera:
    
    tt, xx = mi_funcion_sen( vmax = 1, dc = 0, ff = 1, ph=0, nn = N, fs = fs)

BONUS: 
    
    Implementar alguna otra señal propia de un generador de señales. 


"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdsmodulos as pds

#%% VARIABLES GLOBALES:
    
N = 1000   # cantidad de muestras
fs = N # frecuencia de muestreo (Hz)

#%% DEFINICION DE FUNCION:
"""
    Parámetros:
        vmax: amplitud máxima (Volts)
        dc: valor medio (Volts)
        ff: frecuencia (Hz)
        ph: fase (radianes)
        nn: cantidad de muestras
        fs: frecuencia de muestreo (Hz)
"""
# SENOIDAL: 
"""
    Generador de señal senoidal.
    
    Retorna:
        tt: vector de tiempo
        xx: señal senoidal
"""
    
def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=N , fs=fs):

    # grilla de sampleo temporal
    tt = np.arange(0, nn) / fs

    xx = vmax * np.sin(tt*2*np.pi*ff + ph) + dc

    return tt,xx

# BONUS:
# CUADRADA:
"""
    Generador de señal cuadrada.
    
    Retorna:
        tt: vector de tiempo
        xx: señal cuadrada
        
    NOTA: la funcion np.sign() devuelve el elemento es mayor que cero, 
    devuelve 1; si es menor que cero, devuelve -1; y si el elemento es cero, 
    devuelve 0. 
"""
def mi_funcion_cuad(vmax=1, dc=0, ff=1, ph=0, nn=N, fs=fs):

    tt = np.arange(0, nn) / fs
    xx = dc + vmax * np.sign(np.sin(2 * np.pi * ff * tt + ph))
    return tt, xx

# TRIANGULAR:
"""
    Generador de señal triangular.
    
    Retorna:
        tt: vector de tiempo
        xx: señal triangular
        
    NOTA: 
    identidad trigonometrica de la onda triangular: tri(t) = π/2 * arcsin(sin(t))
    
    sin(x): oscila entre [-1, 1].

    arcsin(sin(x)): "dobla" esa onda, porque el arcsin siempre devuelve 
    valores en [-π/2, π/2].

    Ese doblado da justamente la forma de triángulo.

    Al multiplicar por 2/π, lo normalizamos para que vaya de -1 a +1.

    Después se escala con vmax y se le suma dc para darle amplitud y offset
"""
def mi_funcion_trian(vmax=1, dc=0, ff=1, ph=0, nn=N, fs=fs):

    tt = np.arange(0, nn) / fs
    xx = dc + (2*vmax/np.pi) * np.arcsin(np.sin(2 * np.pi * ff * tt + ph))
    return tt, xx

#%% TESTEO DE FUNCION: 
    
tt_sin, sin = mi_funcion_sen(vmax=1, dc=1, ff=10, ph=np.pi/4, nn=N, fs=fs)

# Armo grafico senoidal
plt.figure()
plt.plot(tt_sin, sin)
plt.title("Señal senoidal generada")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)

tt_sq, sq = mi_funcion_cuad(vmax=1, dc=1, ff=10, ph=np.pi/4, nn=N, fs=fs)

# Armo grafico cuadrada
plt.figure()
plt.plot(tt_sq, sq)
plt.title("Señal cuadrada generada")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)

tt_tri, tria = mi_funcion_trian(vmax=1, dc=1, ff=10, ph=np.pi/4, nn=N, fs=fs)

# Armo grafico triangular 
plt.figure()
plt.plot(tt_tri, tria)
plt.title("Señal triangular generada")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)

# Muestro todos los graficos 
plt.show()
        

    

