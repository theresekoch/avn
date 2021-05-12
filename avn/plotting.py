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
    """
    Generates spectrogram information for plotting

    Parameters
    ----------
    song : avn.dataloading.SongFile object
        SongFile object corresponding to the file to be plotted

    Returns
    -------
    spectrogram_db : numpy ndarray, 2D
        Array containing spectrogram data for plotting. 

    """
    #generate spectrogram
    spectrogram = librosa.stft(song.data, n_fft = 512)
    #log10 scale intensity values of spectrogram
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))
    return spectrogram_db

def plot_spectrogram(spectrogram, sample_rate, ax = None, figsize = (20, 5)):
    """
    Plots a spectrogram of a song. 

    Parameters
    ----------
    spectrogram : numpy ndarray, 2D
        Array containing spectrogram data. 
    sample_rate : int
        Sample rate of audio. Necessary to determine time along the x-axis. 
    ax: matplotlob.axes._subplots.AxesSubplot object
        Axis object must be specified if you want to plot the spectrogram as a 
        subplot within a matplotlib.pyplot figure with other subplots as well. 
        If plotting a spectrogram alone, ax doesn't need to be specified.
    figsize : tuple of floats, optional
        Specifies the dimensions of the output plot. The default is (20, 5).

    Returns
    -------
    None.

    """
    #Create plot with given dimensions
    plt.figure(figsize = figsize, facecolor = 'white')
    #plot spectrogram
    img = librosa.display.specshow(spectrogram, sr = sample_rate, 
                             hop_length = 512 / 4, 
                             x_axis = 'time', 
                             y_axis = 'hz', 
                             cmap = 'viridis', 
                             ax = ax)
    
    