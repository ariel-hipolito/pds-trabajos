#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 09:35:06 2025

@author: ariel
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
import scipy.io as sio
import pandas as pd

from pydub import AudioSegment
from scipy.signal import iirdesign, freqz_sos, sosfiltfilt

#%% LECTURA DE SEÑALES

# ECG
fs_ecg = 1000  # Hz

ecg_no_noise = np.load('ecg_sin_ruido.npy')

# para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

#.squeeze() elimina dimensiones de tamaño 1 (por ejemplo, convierte (1, N) o (N, 1) en (N,)).
ecg_one_lead = mat_struct['ecg_lead'].squeeze()
N = len(ecg_one_lead)

'''
hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']

plt.figure()
plt.plot(ecg_one_lead[5000:12000])

plt.figure()
plt.plot(hb_1)

plt.figure()
plt.plot(hb_2)
'''

ff_ecg, psd_ecg = sig.welch(ecg_one_lead, fs=fs_ecg, window='hann', nperseg = N//40)

plt.figure()
psd_ecg_db = 10 * np.log10(psd_ecg)
plt.plot(ff_ecg, psd_ecg_db)
plt.title("PSD - ECG (Welch)")

plt.xlabel("Frecuencia [Hz]"); plt.ylabel("PSD [dB]")
plt.grid(True)

nyq = fs_ecg / 2.0

#%% PLANTILLA:
    
# Especificaciones: divido por dos porque voy a pasar 2 veces por el filtro
ripple = 1/2  # dB de rizado en banda pasante
atenuacion = 40/2 # dB de atenuación mínima en banda de stop

# Frecuencias críticas (Hz)
ws1 = 0.1   # stopband inferior
wp1 = 0.8   # passband inferior
wp2 = 35.0  # passband superior
ws2 = 40.0  # stopband superior

# Normalización (a Nyquist)
wp = [wp1/nyq, wp2/nyq]
ws = [ws1/nyq, ws2/nyq]

#Aprox modulo: 
f_aprox = ['butter', 'cheby1', 'cheby2', 'cauer']

#%% Diseño con iirdesign
sos_butter = iirdesign(
    wp=wp, ws=ws,
    gpass=ripple, gstop=atenuacion,
    analog=False,
    ftype= f_aprox[0],
    output='sos'
)

sos_cheby1 = iirdesign(
    wp=wp, ws=ws,
    gpass=ripple, gstop=atenuacion,
    analog=False,
    ftype= f_aprox[1],
    output='sos'
)

sos_cheby2 = iirdesign(
    wp=wp, ws=ws,
    gpass=ripple, gstop=atenuacion,
    analog=False,
    ftype= f_aprox[2],
    output='sos'
)

sos_cauer = iirdesign(
    wp=wp, ws=ws,
    gpass=ripple, gstop=atenuacion,
    analog=False,
    ftype= f_aprox[3],
    output='sos'
)

#%% Respuesta en frecuencia iir:
    
w_butter, h_butter = freqz_sos(sos_butter, worN=np.logspace(-2, 1.9, 1000), fs=fs_ecg)

h_but_db = 20 * np.log10(np.abs(h_butter))

w_ch1, h_ch1 = freqz_sos(sos_cheby1, worN=np.logspace(-2, 1.9, 1000), fs=fs_ecg)
h_ch1_db = 20 * np.log10(np.abs(h_ch1))

w_ch2, h_ch2 = freqz_sos(sos_cheby2, worN=np.logspace(-2, 1.9, 1000), fs=fs_ecg)
h_ch2_db = 20 * np.log10(np.abs(h_ch2))

w_cauer, h_cauer = freqz_sos(sos_cauer, worN=np.logspace(-2, 1.9, 1000), fs=fs_ecg)
h_cauer_db = 20 * np.log10(np.abs(h_cauer))

#%% VISUALIZACIÓN: Filtro Cauer con plantilla de diseño iir
plt.figure(figsize=(10, 5))

# Curva del filtro
plt.plot(w_cauer, h_cauer_db, 'k', linewidth=1.3, label='Cauer (SOS)')

# Líneas horizontales de la plantilla
plt.axhline(-ripple, color='green', linestyle='--', linewidth=1, label='Límite banda de paso (0.5 dB)')
plt.axhline(-atenuacion, color='red', linestyle='--', linewidth=1, label='Límite banda de stop (20 dB)')

# Bandas coloreadas
plt.axvspan(0, ws1, color='red', alpha=0.1, label='Stop baja')
plt.axvspan(wp1, wp2, color='green', alpha=0.1, label='Banda de paso')
plt.axvspan(ws2, nyq, color='red', alpha=0.1, label='Stop alta')

# Líneas verticales punteadas
plt.axvline(ws1, color='r', linestyle=':', linewidth=1)
plt.axvline(wp1, color='g', linestyle=':', linewidth=1)
plt.axvline(wp2, color='g', linestyle=':', linewidth=1)
plt.axvline(ws2, color='r', linestyle=':', linewidth=1)

# Ejes y estética
plt.title('Filtro Cauer con plantilla de diseño', fontsize=12)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.xlim(0, 80)
plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
plt.legend(loc='lower center', ncol=3, fontsize=8)
plt.tight_layout()
plt.show()


#%% DIAGRAMA POLOS Y CEROS (PLANO-Z)

from scipy.signal import sos2tf, tf2zpk

# Obtengo b,a del sos (producto de secciones)
b, a = sos2tf(sos_butter)           # devuelve coeficientes del numerador y denominador global
z, p, k = tf2zpk(b, a)      # ceros (z), polos (p), ganancia (k)

# Configuro figura
plt.figure(figsize=(6,6))
ax = plt.gca()
ax.set_title('Diagrama de polos y ceros (plano-z)')
ax.set_xlabel('Parte real')
ax.set_ylabel('Parte imaginaria')

# Círculo unidad
theta = np.linspace(0, 2*np.pi, 400)
plt.plot(np.cos(theta), np.sin(theta), color='lightgray', linestyle='--', linewidth=1)

# Ejes
plt.axhline(0, color='gray', linewidth=0.8)
plt.axvline(0, color='gray', linewidth=0.8)

# Ploteo ceros y polos globales
if len(z) > 0:
    plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='b', s=80, label='Ceros')
plt.scatter(np.real(p), np.imag(p), marker='x', color='r', s=80, label='Polos')

for i, section in enumerate(sos_butter):
    b_s = section[:3]
    a_s = section[3:]
    z_s, p_s, k_s = tf2zpk(b_s, a_s)
    if len(z_s) > 0:
        plt.scatter(np.real(z_s), np.imag(z_s), marker='o', facecolors='none', edgecolors='cyan', s=30, alpha=0.7, zorder=1)
    plt.scatter(np.real(p_s), np.imag(p_s), marker='x', color='magenta', s=30, alpha=0.7, zorder=1)

# Ajustes de escala y leyenda
lim = 1.2
plt.xlim([-lim, lim])
plt.ylim([-lim, lim])
plt.gca().set_aspect('equal', 'box')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.show()

# Comprobación de estabilidad
inside = np.sum(np.abs(p) < 1.0)
total = len(p)
print(f"Polos dentro del círculo unidad: {inside}/{total} -> {'Estable' if inside==total else 'Inestable o aproximación numérica'}")
    
# %% FIR

from scipy.signal import firwin2, freqz, firls, lfilter

# Cantidad de coeficientes (cuanto mayor, más preciso y más transitorio)
taps = 9001  

# Normalización de frecuencias
f = np.array([0, ws1+0.18, wp1, wp2, ws2-0.18, nyq]) / nyq
#g_deseada = np.array([0, 0, 1, 1, 0, 0])
gain_db = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion]) 
g_deseada = 10**(gain_db / 20)

# Diseño FIR con firwin2
b_fir_win = firwin2(taps, f, g_deseada, window='hamming')

# Respuesta en frecuencia del FIR
w_fir, h_fir_win = freqz(b_fir_win, worN=np.logspace(-2, 1.9, 1000), fs=fs_ecg)
h_fir_win_db = 20 * np.log10(np.abs(h_fir_win) + 1e-10)

#%% VISUALIZACIÓN: FIR con plantilla de diseño
plt.figure(figsize=(10,5))
plt.plot(w_fir, h_fir_win_db, 'k', linewidth=1.3, label='FIR (firwin2)')

plt.axhline(-ripple, color='green', linestyle='--', linewidth=1, label='Límite banda de paso (0.5 dB)')
plt.axhline(-atenuacion, color='red', linestyle='--', linewidth=1, label='Límite banda de stop (20 dB)')

plt.axvspan(0, ws1, color='red', alpha=0.1, label='Stop baja')
plt.axvspan(wp1, wp2, color='green', alpha=0.1, label='Banda de paso')
plt.axvspan(ws2, nyq, color='red', alpha=0.1, label='Stop alta')

plt.axvline(ws1, color='r', linestyle=':', linewidth=1)
plt.axvline(wp1, color='g', linestyle=':', linewidth=1)
plt.axvline(wp2, color='g', linestyle=':', linewidth=1)
plt.axvline(ws2, color='r', linestyle=':', linewidth=1)

plt.title('Filtro FIR (firwin2) con plantilla de diseño')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.xlim(0, 80)
plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
plt.legend(loc='lower center', ncol=3, fontsize=8)
plt.tight_layout()
plt.show()


# %% CUADRADOS MINIMOS

numtaps_ls = 3001  

# Frecuencias normalizadas (0 a 1 → Nyquist = 1)
f_ls = [
    0,
    ws1+0.15,    # fin stop baja
    wp1,    # inicio pasabanda
    wp2,    # fin pasabanda
    ws2-4.5,    # inicio stop alta
    nyq
]

desired = np.array([0, 0, 1, 1, 0, 0])

# Diseño del FIR
b_fir_ls = firls(numtaps_ls, f_ls, desired, weight=None, fs=fs_ecg)

# Respuesta en frecuencia
w_ls, h_ls = freqz(b_fir_ls, worN=np.logspace(-2, 1.9, 1000), fs=fs_ecg)
h_ls_db = 20 * np.log10(np.abs(h_ls) + 1e-10)

#%% VISUALIZACIÓN: FIR (Mínimos cuadrados)
plt.figure(figsize=(10,5))
plt.plot(w_ls, h_ls_db, 'k', linewidth=1.3, label='FIR (Mínimos Cuadrados)')

plt.axhline(-ripple, color='green', linestyle='--', linewidth=1, label='Límite banda de paso')
plt.axhline(-atenuacion, color='red', linestyle='--', linewidth=1, label='Límite banda de stop')

plt.axvspan(0, ws1, color='red', alpha=0.1, label='Stop baja')
plt.axvspan(wp1, wp2, color='green', alpha=0.1, label='Banda de paso')
plt.axvspan(ws2, nyq, color='red', alpha=0.1, label='Stop alta')

plt.title('Filtro FIR (Mínimos Cuadrados - firls)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.xlim(0, 80)
plt.grid(True, which='both', linestyle='--', linewidth=0.6)
plt.legend(loc='lower center', ncol=3, fontsize=8)
plt.tight_layout()
plt.show()

#%%
# Filtrado doble: pasa dos veces por el mismo filtro 
# COMPARACION FILTROS FIR: VENTANAS VS CUADRADOS MINIMOS

ecg_fir_ls = lfilter(b_fir_ls, 1.0, lfilter(b_fir_ls, 1.0, ecg_one_lead))
retardo_ls = (len(b_fir_ls) - 1) // 2

ecg_fir_win = lfilter(b_fir_win, 1.0, lfilter(b_fir_win , 1.0, ecg_one_lead))
retardo_win = (len(b_fir_win) - 1) // 2

plt.figure(figsize=(12,5))
plt.plot(ecg_one_lead, color='gray', alpha=0.45, label='ECG con ruido', zorder=1)
plt.plot(ecg_fir_win[retardo_win:], color='red', linewidth=1.4, label='FIR (Ventanas)', zorder=3)
plt.plot(ecg_fir_ls[retardo_ls:],  color='blue',  linewidth=1.4, label='FIR (Mínimos Cuadrados)', zorder=2)

plt.title('Comparación ECG con ruido: FIR Ventanas vs FIR Cuadrados Mínimos')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

#%% COMPARACION FILTROS IIR: 

ecg_filt_cheby2 = sosfiltfilt(sos_cheby2, ecg_one_lead)
ecg_filt_cauer = sosfiltfilt(sos_cauer, ecg_one_lead)

plt.figure(figsize=(10,4))
plt.plot(ecg_one_lead, color='gray', alpha=0.4, label='ECG original')
plt.plot(ecg_filt_cheby2, color='red', linewidth=1.4, label='ECG filtrado Cheby2')
plt.plot(ecg_filt_cauer, color='blue',  linewidth=1.4, label='ECG filtrado Cauer')

plt.title('Comparación ECG con ruido: IIR Cheby2 vs IIR Cauer')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()

#%%

#################################
# Regiones de interés sin ruido #
#################################

cant_muestras = len(ecg_one_lead)

regs_interes = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_filt_cauer[zoom_region], label='Cauer')
    plt.plot(zoom_region, ecg_fir_win[zoom_region + retardo_win], label='FIR Window')
   
    plt.title('ECG sin ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
 
#################################
# Regiones de interés con ruido #
#################################
 
regs_interes = (
        np.array([5, 5.2]) *60*fs_ecg, # minutos a muestras
        np.array([12, 12.4]) *60*fs_ecg, # minutos a muestras
        np.array([15, 15.2]) *60*fs_ecg, # minutos a muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_filt_cheby2[zoom_region], label='Cheby2')
    plt.plot(zoom_region, ecg_fir_win[zoom_region + retardo_win], label='FIR Window')
   
    plt.title('ECG con ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
    
# %% COMPARACIÓN RESPUESTAS EN FRECUENCIA DE TODOS LOS FILTROS

# Grilla de frecuencias personalizada (rad/muestra)
w_rad = np.append(np.logspace(-3, 0.8, 1000), np.logspace(0.9, 1.8, 1000))
w_rad = np.append(w_rad, np.linspace(64, nyq, 1000, endpoint=True)) / nyq * np.pi

# Evaluación de cada filtro con freqz/freqz_sos
w_butter, h_butter = freqz_sos(sos_butter, worN=w_rad, fs=fs_ecg)
w_ch1, h_ch1       = freqz_sos(sos_cheby1, worN=w_rad, fs=fs_ecg)
w_ch2, h_ch2       = freqz_sos(sos_cheby2, worN=w_rad, fs=fs_ecg)
w_cauer, h_cauer   = freqz_sos(sos_cauer, worN=w_rad, fs=fs_ecg)

w_fir_win, h_fir_win = freqz(b_fir_win, worN=w_rad, fs=fs_ecg)
w_fir_ls,  h_fir_ls  = freqz(b_fir_ls,  worN=w_rad, fs=fs_ecg)

# Conversión a dB
h_but_db   = 20*np.log10(np.abs(h_butter) + 1e-10)
h_ch1_db   = 20*np.log10(np.abs(h_ch1) + 1e-10)
h_ch2_db   = 20*np.log10(np.abs(h_ch2) + 1e-10)
h_cauer_db = 20*np.log10(np.abs(h_cauer) + 1e-10)
h_win_db   = 20*np.log10(np.abs(h_fir_win) + 1e-10)
h_ls_db    = 20*np.log10(np.abs(h_fir_ls) + 1e-10)

# Gráfico comparativo
plt.figure(figsize=(12,6))
plt.plot(w_butter, h_but_db,   label='Butterworth', color='black')
plt.plot(w_ch1,    h_ch1_db,   label='Chebyshev I', color='orange')
plt.plot(w_ch2,    h_ch2_db,   label='Chebyshev II', color='red')
plt.plot(w_cauer,  h_cauer_db, label='Cauer (Elíptico)', color='blue')
plt.plot(w_fir_win,h_win_db,   label='FIR Ventanas', color='green')
plt.plot(w_fir_ls, h_ls_db,    label='FIR Mínimos Cuadrados', color='purple')

# Líneas de la plantilla
plt.axhline(-ripple,     color='green', linestyle='--', linewidth=1, label='Límite pasabanda (-0.5 dB)')
plt.axhline(-atenuacion, color='red',   linestyle='--', linewidth=1, label='Límite stopband (-20 dB)')

# Estética
plt.title('Comparación de respuestas en frecuencia de filtros IIR y FIR')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.xlim(0, 80)
plt.ylim(-80, 5)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower center', ncol=3, fontsize=8)
plt.tight_layout()
plt.show()
