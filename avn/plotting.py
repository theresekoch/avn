# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:39:10 2021

@author: Therese
"""
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def make_spectrogram(song):
    spectrogram = librosa.stft(song.data, n_fft = 512)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))
    return spectrogram_db

def plot_spectrogram(spectrogram, sample_rate, figsize = (20, 5)):
    plt.figure(figsize = figsize, facecolor = 'white')
    
    librosa.display.specshow(spectrogram, sr = sample_rate, 
                             hop_length = 512 / 4, 
                             x_axis = 'time', 
                             y_axis = 'hz', 
                             cmap = 'viridis')
    