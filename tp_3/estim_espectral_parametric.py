#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Spyder(Phyton 3.10)

@author: Ariel Hipolito

Descripción:
    

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
import scipy.io as sio
import pandas as pd

from pydub import AudioSegment
#%% LECTURA DE SEÑALES

# ECG y PPG
ecg_one_lead = np.load('ecg_sin_ruido.npy')
fs_ecg = 1000  # Hz

ppg = np.load('ppg_sin_ruido.npy')
fs_ppg = 400  # Hz

# Audios
fs_audio_A, wav_data_A = sio.wavfile.read('la cucaracha.wav')
fs_audio_B, wav_data_B = sio.wavfile.read('prueba psd.wav')
fs_audio_C, wav_data_C = sio.wavfile.read('silbido.wav')


#%% FUNCIÓN PARA OBTENER ANCHO DE BANDA EFECTIVO

def obtener_bw(psd, ff, porcentaje=0.9):
    """Calcula el ancho de banda que contiene el porcentaje indicado
    de la potencia total de la señal (por defecto, 90%)."""
    df = ff[1] - ff[0]
    pot_total = np.sum(psd) * df
    pot_acum = np.cumsum(psd) * df
    idx = np.where(pot_acum >= pot_total * porcentaje)[0][0]
    return ff[idx]


#%% ESTIMACIÓN DE PSD - MÉTODO DE WELCH

# ECG y PPG (señales cortas, x=10)
ff_ecg, psd_ecg = sig.welch(ecg_one_lead, fs=fs_ecg, window='hann', nperseg=int(len(ecg_one_lead)/10))
ff_ppg, psd_ppg = sig.welch(ppg, fs=fs_ppg, window='hann', nperseg=int(len(ppg)/10))

# Audios (señales largas → x=40 para suavizar)
ff_A, psd_A = sig.welch(wav_data_A, fs=fs_audio_A, window='hann', nperseg=int(len(wav_data_A)/40))
ff_B, psd_B = sig.welch(wav_data_B, fs=fs_audio_B, window='hann', nperseg=int(len(wav_data_B)/40))
ff_C, psd_C = sig.welch(wav_data_C, fs=fs_audio_C, window='hann', nperseg=int(len(wav_data_C)/40))


#%% CÁLCULO DE ANCHO DE BANDA (90 % DE POTENCIA)

bw_ecg = obtener_bw(psd_ecg, ff_ecg)
bw_ppg = obtener_bw(psd_ppg, ff_ppg)
bw_A = obtener_bw(psd_A, ff_A)
bw_B = obtener_bw(psd_B, ff_B)
bw_C = obtener_bw(psd_C, ff_C)

# Tabla resumen
tabla_bw = pd.DataFrame({
    'Señal': ['ECG', 'PPG', 'Audio A (La Cucaracha)', 'Audio B (Prueba)', 'Audio C (Silbido)'],
    'Frecuencia de muestreo [Hz]': [fs_ecg, fs_ppg, fs_audio_A, fs_audio_B, fs_audio_C],
    'BW (90% Potencia) [Hz]': [bw_ecg, bw_ppg, bw_A, bw_B, bw_C]
})
print("\n=== ANCHO DE BANDA EFECTIVO (90% POTENCIA) ===\n")
print(tabla_bw.to_string(index=False))


#%% GRAFICADO DE PSD (ECG, PPG, AUDIOS)

plt.figure(figsize=(18, 6))

# ECG
plt.subplot(2, 3, 1)
psd_ecg_db = 10 * np.log10(psd_ecg / np.max(psd_ecg) + 1e-12)
plt.plot(ff_ecg, psd_ecg_db)
plt.title("PSD - ECG (Welch)")
plt.axvline(bw_ecg, color='green', linestyle='--', label=f'BW 90% = {bw_ecg:.1f} Hz')
plt.xlabel("Frecuencia [Hz]"); plt.ylabel("PSD [dB]")
plt.ylim([-80, 5]); plt.legend(); plt.grid(True)

# PPG
plt.subplot(2, 3, 2)
psd_ppg_db = 10 * np.log10(psd_ppg / np.max(psd_ppg) + 1e-12)
plt.plot(ff_ppg, psd_ppg_db)
plt.title("PSD - PPG (Welch)")
plt.axvline(bw_ppg, color='green', linestyle='--', label=f'BW 90% = {bw_ppg:.1f} Hz')
plt.xlabel("Frecuencia [Hz]"); plt.ylabel("PSD [dB]")
plt.ylim([-80, 5]); plt.legend(); plt.grid(True)

# Audio A
plt.subplot(2, 3, 3)
psd_A_db = 10 * np.log10(psd_A / np.max(psd_A) + 1e-12)
plt.plot(ff_A, psd_A_db)
plt.title("PSD - Audio A (La Cucaracha)")
plt.axvline(bw_A, color='green', linestyle='--', label=f'BW 90% = {bw_A:.1f} Hz')
plt.xlabel("Frecuencia [Hz]"); plt.ylabel("PSD [dB]")
plt.xlim([0, 8000]); plt.ylim([-80, 5])
plt.legend(); plt.grid(True)

# Audio B
plt.subplot(2, 3, 4)
psd_B_db = 10 * np.log10(psd_B / np.max(psd_B) + 1e-12)
plt.plot(ff_B, psd_B_db)
plt.title("PSD - Audio B (Prueba)")
plt.axvline(bw_B, color='green', linestyle='--', label=f'BW 90% = {bw_B:.1f} Hz')
plt.xlabel("Frecuencia [Hz]"); plt.ylabel("PSD [dB]")
plt.xlim([0, 8000]); plt.ylim([-80, 5])
plt.legend(); plt.grid(True)

# Audio C
plt.subplot(2, 3, 5)
psd_C_db = 10 * np.log10(psd_C / np.max(psd_C) + 1e-12)
plt.plot(ff_C, psd_C_db)
plt.title("PSD - Audio C (Silbido)")
plt.axvline(bw_C, color='green', linestyle='--', label=f'BW 90% = {bw_C:.1f} Hz')
plt.xlabel("Frecuencia [Hz]"); plt.ylabel("PSD [dB]")
plt.xlim([0, 8000]); plt.ylim([-80, 5])
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()

#%% BONUS: SEÑAL PROPUESTA (SENO MODULADO)

# Cargar archivo MP3
audio = AudioSegment.from_mp3("whale_sound.mp3")

# Exportar a WAV
audio.export("whale_sound.wav", format="wav")

fs_ballena, senal_ballena = sio.wavfile.read('whale_sound.wav')

# Si es estéreo, convertir a mono tomando un canal
if senal_ballena.ndim > 1:
    senal_ballena = senal_ballena[:, 0]

# --- Estimación de PSD usando método de Welch ---
nperseg = int(len(senal_ballena) / 20)  # podés ajustar el denominador si querés
ff_ballena, psd_ballena = sig.welch(senal_ballena, fs=fs_ballena, window='hann', nperseg=nperseg)

# --- Cálculo del ancho de banda ---
bw_ballena = obtener_bw(psd_ballena, ff_ballena)
print(f"\nAncho de banda del sonido de ballena (90% potencia): {bw_ballena:.2f} Hz")

# --- Gráfico del espectro ---
plt.figure(figsize=(10, 4))
plt.plot(ff_ballena, 10 * np.log10(psd_ballena + 1e-12))
plt.axvline(bw_ballena, color='green', linestyle='--', label=f'BW 90% = {bw_ballena:.1f} Hz')
plt.title("PSD - Sonido de ballena (MP3 convertido a WAV)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [dB/Hz]")
plt.legend()
plt.grid(True)
plt.show()

'''
fs_sint = 8000
t = np.arange(0, 2, 1/fs_sint)
senal_bonus = (1 + 0.5*np.sin(2*np.pi*2*t)) * np.sin(2*np.pi*440*t)  # AM de 2 Hz

ff_bonus, psd_bonus = sig.welch(senal_bonus, fs=fs_sint, nperseg=1024)
bw_bonus = obtener_bw(psd_bonus, ff_bonus)

print(f"\nAncho de banda de señal sintética (90% potencia): {bw_bonus:.2f} Hz")

plt.figure(figsize=(10, 4))
psd_bonus_db = 10 * np.log10(psd_bonus / np.max(psd_bonus) + 1e-12)
plt.plot(ff_bonus, psd_bonus_db)
plt.axvline(bw_bonus, color='green', linestyle='--', label=f'BW 90% = {bw_bonus:.1f} Hz')
plt.title("PSD - Señal Sintética (Tono 440 Hz modulada en amplitud)")
plt.xlabel("Frecuencia [Hz]"); plt.ylabel("PSD [dB, relativo]")
plt.legend(); plt.grid(True)
plt.show()
'''