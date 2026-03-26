"""
Created on Wed May  8 09:44:27 2024

@author: theja
"""

import scipy.signal as signal 
import numpy as np 
import matplotlib.pyplot as plt 
import sounddevice as sd
import soundfile as sf

#%%
# make a sweep
durns = np.array([3, 4, 5, 8, 10] )*1e-3
fs = 192000 # Hz

all_sweeps = []
for durn in durns:
    t = np.linspace(0, durn, int(fs*durn))
    start_f, end_f = 100, 24e3
    sweep = signal.chirp(t, start_f, t[-1], end_f)
    sweep *= signal.windows.tukey(sweep.size, 0.2)
    sweep *= 0.8
    sweep_padded = np.pad(sweep, pad_width=[int(fs*0.1)]*2, constant_values=[0,0])
    all_sweeps.append(sweep_padded)
    
sweeps_combined = np.concatenate(all_sweeps)

target_duration = 10 * 60  # 10 minutes in seconds

current_duration = len(sweeps_combined) / fs

# how many times to repeat
repeats = int(np.ceil(target_duration / current_duration))

# repeat signal
long_signal = np.tile(sweeps_combined, repeats)

# trim exactly to 10 minutes
long_signal = long_signal[:int(fs * target_duration)]

sf.write('soundGenerate/01_24k_5sweeps.wav', sweeps_combined, samplerate=fs)
sf.write('soundGenerate/01_24k_10min.wav', long_signal, samplerate=fs)