# -*- coding: utf-8 -*-
"""
Created on Thu Mar 2 09:18:15 2023

@author: Therese
"""

import librosa
import librosa.display
import numpy as np
import avn.dataloading
import avn.plotting
import glob
import matplotlib.pyplot as plt
import scipy.signal
import math
import pandas as pd
import re
import sklearn
import seaborn as sns
import os
import datetime

class SongInterval:

    def __init__(self, song_file, onset = 0, offset = None,
                 win_length = 400, hop_length = 40, n_fft = 1024, 
                 max_F0 = 1830, min_frequency = 380, freq_range = 0.5, 
                 baseline_amp = 70, fmax_yin = 8000, ):
        
        #set specified or default parameters
        self.onset = onset
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.max_F0 = max_F0
        self.min_frequency = min_frequency
        self.freq_range = freq_range
        self.baseline_amp = baseline_amp
        self.fmax_yin = fmax_yin

        #if offset timestamp is not specified, the end of the file is set as the offset
        if offset is None:
            self.offset = song_file.duration
        else: 
            self.offset = offset

        self.duration = self.offset - self.onset

        #select the song data between the specified onset and offset
        self.song_data, __, __ = avn.dataloading.Utils.select_syll(song_file, self.onset, self.offset)
        #rescale song data between -1 and 1
        self.song_data = self.song_data / (2**15)
        self.sample_rate = song_file.sample_rate
        self.full_file = song_file

        #initialize empty attributes to be calculated as needed
        self.windows = None
        self.cepstra = None
        self.windows1 = None
        self.windows2 = None
        self.spectra1 = None
        self.spectra2 = None
        self.power_spectra = None
        self.min_freq_index = None
        self.max_freq_index = None
        self.time_derivs = None
        self.freq_derivs = None
        self.goodness = None
        self.mean_frequency = None
        self.FM = None
        self.AM = None
        self.entropy = None
        self.amplitude = None
        self.pitch = None

    def calc_goodness(self):
        """Calculates the goodness of pitch of each window in a song interval.

        Goodness of pitch is an estimate of the harmonic periodicity of a signal. 
        Higher values indicate a more periodic sound (like a harmonic stack), whereas 
        lower values indicate less periodic sounds (like noise). Formally, it is the 
        peak of the cepstrum of the signal for harmonics with a fundamental frequency
        below `max_F0`. 

        Returns:
            np.array: array containing the goodness of pitch for each frame in the song interval.
        """
        if self.cepstra is None:
            self._get_cepstra()

        #calculate cutoff index to exclude any harmonics above max_f0 Hz
        quefrencies = np.array(range(self.win_length))/self.sample_rate
        quefrency_cutoff = 1 / self.max_F0
        cutoff_idx = np.min(np.argwhere(quefrencies > quefrency_cutoff))-1

        #initialize empty array to store values
        goodnesses = np.zeros(len(self.windows))
        #loop over each window in the song interval
        for i in range(len(self.windows)):
            cepstrum = self.cepstra[i, :]
            #goodness is the height of the highest peak in the first half of the cepstrum, above the cutoff idx
            goodness = np.max(cepstrum[cutoff_idx:int(np.floor(len(cepstrum)/2)-1)])
            goodnesses[i] = goodness

        self.goodness = goodnesses
        return self.goodness 
    
    def calc_mean_frequency(self):
        """Calculates the mean frequency of each window in a song interval.
        This is one way to estimate the pitch of a signal. It is the center of the distribution of power across frequencies in the signal. 
        For another estimate of pitch, see `SongInterval.calc_pitch()`
        Returns:
            np.array: array containing the mean frequency of each frame in the song interval.
        """
        if self.power_spectra is None:
            self._get_power_spectra()

        if self.max_freq_index is None:
            self._get_freq_indices()
        
        #create vector of frequencies in Hz for each frequency band in the power spectrum
        frequencies = np.arange(1, self.max_freq_index + 1) * self.sample_rate / self.n_fft

        #initialize empty array to store values
        mean_frequencies = np.zeros(len(self.windows))

        #loop over each window in the song interval
        for i in range(len(self.windows)):
            power_spectrum = self.power_spectra[i]
            ##calculate mean frequency for each frame, considering only frequencies above min_freq_index and below max_freq_index
            power_spectrum = power_spectrum[ self.min_freq_index : self.max_freq_index ]
            mean_frequency = np.sum(power_spectrum * frequencies[self.min_freq_index:]) / np.sum(power_spectrum)
            mean_frequencies[i] = mean_frequency

        self.mean_frequency = mean_frequencies
        return self.mean_frequency

    def calc_frequency_modulation(self):
        """Calculates the frequency modulation of each window in a song interval.

        Frequency Modulation can be thought of as the slope of frequency traces in a spectrogram. A high
        frequency modulation score is indicative of a sound who's pitch is changing rapidly, or which is 
        noisy and has an unstable pitch. A low frequency modulation score indicates that the pitch of a 
        sound is stable (like in a flat harmonic stack). This implementation is based on SAP. 

        Returns:
            np.array: array containing the mean frequency of each frame in the song interval.
        """
        if self.freq_derivs is None:
            self._get_freq_derivative()
        if self.time_derivs is None:
            self._get_time_derivative()

        #get small number to avoid potential divide by zero errors
        eps = np.finfo(np.double).eps
        
        #initialize empty array to store values
        FMs = np.zeros(len(self.windows))
        #loop over each window in the song interval
        for i in range(len(self.windows)):
            #calculate the frequency modulation of current frame
            FM = np.arctan(np.max(self.time_derivs[i]) / (np.max(self.freq_derivs[i]) + eps))
            FMs[i] = FM
        self.FM = FMs
        return self.FM
    
    def calc_amplitude_modulation(self):
        """Calculates the amplitude modulation of each window in a song interval.

        Amplitude modulation is a measure of the rate of change of the amplitude of a signal.
        It will be positive at the beginning of a song syllable and negative at the end. 

        Returns:
            np.array: array containing the amplitude modulation of each frame in the song interval.
        """
        if self.time_derivs is None:
            self._get_time_derivative()

        #initialize an empty array to store values
        AMs = np.zeros(len(self.windows))
        #loop over each window in the song interval
        for i in range(len(self.windows)):
            time_derivative = self.time_derivs[i]
            #calculate amplitude modulation of current frame
            AMs[i] = np.sum(time_derivative)
        self.AM = AMs
        return self.AM

    def calc_entropy(self):
        """Calculates the Wiener entropy of each window in a song interval.

        Weiner entropy is a measure of the uniformity of power spread across frequency bands in a frame of audio. 
        The output of this function is log-scaled Weiner entropy, which can range in value from 0 to negative 
        infinity. A score close to 0 indicates broadly spread power across frequency bands, ie a less structured 
        sound like noise. A large negative score indicates low uniformity across frequency bands, ie a more 
        structured sound like a harmonic stack.

        Returns:
            np.array: array containing the log-scaled Weiner entropy of each frame in the song interval.
        """
        if self.power_spectra is None:
            self._get_power_spectra()
        if self.max_freq_index is None:
            self._get_freq_indices()

        #initialize empty array to store values
        entropies = np.zeros(len(self.windows))
        #loop over each window in the song interval
        for i in range(len(self.windows)):
            #select range of frequencies in power spectrum between min_freq_index and max_freq_index
            power_spectrum = self.power_spectra[i, self.min_freq_index : self.max_freq_index]
            #calculate entropy for current frame
            sum_log = np.sum(np.log(power_spectrum))
            log_sum = np.log(np.sum(power_spectrum)/(self.max_freq_index - self.min_freq_index - 1))#-1 to account for differences in matlab vs. python indexing
            entropy = sum_log / (self.max_freq_index - self.min_freq_index -1) - log_sum
            entropies[i] = entropy
        self.entropy = entropies
        return self.entropy
    
    def calc_amplitude(self):
        """Calculates the amplitude of each window in a song interval.

        Amplitude is the volume of a sound in decibels, considering only frequencies above self.min_frequency.

        Returns:
            np.array: array containing the amplitude of each frame in the song interval in decibels
        """
        if self.power_spectra is None:
            self._get_power_spectra()
        if self.max_freq_index is None:
            self._get_freq_indices()

        amplitudes = np.zeros(len(self.windows))
        for i in range(len(self.windows)):
            #select range of frequencies in power spectrum between min and max freq indices for current window
            power_spectrum = self.power_spectra[i, self.min_freq_index:self.max_freq_index]
            #compute amplitude
            pow_spect_sum = np.sum(power_spectrum)
            #convert amplitude to dB and actor in baseline amplitude
            amplitudes[i] = 10 * np.log10(pow_spect_sum) + self.baseline_amp
        self.amplitude = amplitudes
        return self.amplitude

    def calc_pitch(self):
        """Estimates the fundamental frequency (or pitch) of each window in a song interval using the yin algorithm. 

        For more information on the YIN algorithm for fundamental frequency estimation, please refer to the documentaion 
        for `librosa.yin()`. 

        Returns:
            np.array: array containing the YIN estimated fundamental frequency of each frame in the song interval in Hertz. 
        """
        pitch = librosa.yin(self.song_data, fmin = self.min_frequency, fmax = self.fmax_yin,
                            sr = self.sample_rate, hop_length=self.hop_length)
        self.pitch = pitch
        return pitch

    def _make_windows(self):
        """Slice the song data into overlapping frames. 
        
        This function creates an array of frames, each of length `self.win_length` with the start of each 
        frame separated by `self.hop_length` samples. See `librosa.util.frame()` documentation for more information. 
        """
        wave_padded = np.pad(self.song_data, pad_width= self.win_length//2)
        song_frames = librosa.util.frame(wave_padded, frame_length=self.win_length, hop_length= self.hop_length, axis = 0)
        self.windows = song_frames

    def _apply_tapers(self):
        """Apply 2 multitapers to the song frame. 

        Applying multitapers to song frame before calculating its spectrum through 
        a fast fourier transform (FFT) can improve the FFT's estimate of the power
        spectrum for emperical data. 
        """
        if self.windows is None:
            self._make_windows()

        tapers = scipy.signal.windows.dpss(self.win_length, 1.5, Kmax = 2)
        self.windows1 = self.windows * tapers[0, :]
        self.windows2 = self.windows * tapers[1, :]

    def _get_spectra(self):
        """Calculate the complex spectrum of each window in the song interval

        Calculates the complex spectrum (meaning it contains both real and imaginary components)
        for each window with either taper 1 or taper 2 applied. These are necessary for computing 
        amplitude and frequency modulation. 
        """
        if self.windows1 is None:
            self._apply_tapers()
        #initialize empty arrays to hold spectra
        spectra1 = np.zeros((len(self.windows), self.n_fft), dtype = complex)
        spectra2 = np.zeros((len(self.windows), self.n_fft), dtype = complex)
        #compute spectrum for each window
        for i in range(len(self.windows)):
            spect1 = np.fft.fft(self.windows1[i], n = self.n_fft)
            spect2 = np.fft.fft(self.windows2[i], n = self.n_fft)
            spectra1[i] = spect1
            spectra2[i] = spect2
        self.spectra1 = spectra1
        self.spectra2 = spectra2

    def _get_cepstra(self):
        """Compute the real cepstrum of a frame of audio.

        The cepstrum of a signal is the inverse fourier transform of the log fourier transform
        of the signal. It is useful for looking at periodic patterns across frequency bands in 
        a spectrum and is used to calculate the goodness of pitch of a signal.
        """
        if self.spectra1 is None:
            self._get_spectra()

        #initialize empty array to hold cepstrum for each frame
        cepstra = np.zeros((len(self.windows), self.n_fft))
        #loop over each window in the song interval
        for i in range(len(self.windows)):
            #select only the real component of the spectrum for the current frame
            spectrum = abs(self.spectra1[i, :])
            log_spectrum = np.log(spectrum, out = spectrum, where = spectrum>0)
            #calculate cepstrum for each frame
            cepstrum = np.fft.ifft(log_spectrum, n = self.n_fft).real
            cepstra[i] = cepstrum
        self.cepstra = cepstra

    def _get_power_spectra(self):
        """Computes the power spectrum of each window in a song interval.

        The power spectrum of a signal gives the distribution of power across frequency bands of that
        signal. It is necessary to compute mean frequency, frequency modulation, weiner entropy, etc.

        """
        if self.spectra1 is None:
            self._get_spectra()
        #initialize empty array to hold power spectrum for each frame
        power_spectra = np.zeros((len(self.windows), self.n_fft))
        #calculate power spectrum for each frame:
        for i in range(len(self.windows)):
            spect1 = self.spectra1[i]
            spect2 = self.spectra2[i]
            spectrum = np.abs(spect1) + np.abs(spect2)
            power_spectrum = spectrum ** 2
            power_spectra[i, :] = power_spectrum
        self.power_spectra = power_spectra

    def _get_freq_indices(self):
        """Calculate the index in a spectrum corresponding to the min_frequency frequency band and a maximum 
        frequency band determined by `freq_range`. 

        The output min_freq_index and max_freq_index values can be used to select a subset of a spectrum containing
        only relevant frequency bands for the subsequent calculation of various acoustic features such as mean frequency
        and Weiner entropy. 
        """
        # determine upper frequency cutoff for power spectrum to consider - since the real component 
        # is mirrored about the center, we only need to consider the first half. Then we may want 
        # to restrict further based on freq_range. 
        self.max_freq_index = int(np.floor(self.n_fft * self.freq_range / 2))

        #determine index of spectrum bin corresponding to min_frequency
        frequencies = np.arange(self.max_freq_index) * self.sample_rate / self.n_fft
        self.min_freq_index = np.min(np.argwhere(frequencies > self.min_frequency))

    def _get_time_derivative(self):
        """Calculate the time derivative of a spectrum for each window in a song interval.
        The time derivative is subsequently used to calculate amplitude and frequency modulation. 
        """
        if self.spectra1 is None:
            self._get_spectra()
        if self.max_freq_index is None:
            self._get_freq_indices()

        #initialize empty array to hold derivatives
        time_derivs = np.zeros((len(self.windows), self.max_freq_index))
        #loop over each window in the song interval
        for i in range(len(self.windows)):
            #select only relevant frequencies from spectra for current window
            spect1 = self.spectra1[i, :self.max_freq_index]
            spect2 = self.spectra2[i, :self.max_freq_index]
            #calculate time derivative. 
            time_derivs[i] = (-spect1.real * spect2.real) - (spect1.imag * spect2.imag)
        self.time_derivs = time_derivs

    def _get_freq_derivative(self):
        """Calculate the frequency derivative of a spectrum for each window in a song interval.
        The frequency derivative is subsequently used to calculate amplitude and frequency modulation. 
        """
        if self.spectra1 is None:
            self._get_spectra()
        if self.max_freq_index is None:
            self._get_freq_indices()

        #initialize empty array to hold derivatives
        freq_derivs = np.zeros((len(self.windows), self.max_freq_index))
        #loop over each window in the song interval
        for i in range(len(self.windows)):
            #select only relevant frequencies from spectra for current window
            spect1 = self.spectra1[i, :self.max_freq_index]
            spect2 = self.spectra2[i, :self.max_freq_index]
            #calculate frequency derivative. 
            freq_derivs[i] = (spect1.imag * spect2.real) - (spect1.real * spect2.imag)
        self.freq_derivs = freq_derivs

    def calc_all_features(self):
        """Calculate all acoustic features for each window in a song interval. 

        This method returns a dictionary containing the time series values for 
        Goodness, Mean_frequency, Entropy, Amplitude, Amplitude_modulation, 
        Frequency_modulation, and Pitch calculated for each window in the song interval. 

        Returns:
            dict: dictionary containing the time series values for 
        Goodness, Mean_frequency, Entropy, Amplitude, Amplitude_modulation, 
        Frequency_modulation, and Pitch as np.arrays. 
        """
        if self.goodness is None:
            self.calc_goodness()
        if self.mean_frequency is None:
            self.calc_mean_frequency()
        if self.entropy is None:
            self.calc_entropy()
        if self.amplitude is None:
            self.calc_amplitude()
        if self.AM is None:
            self.calc_amplitude_modulation()
        if self.FM is None:
            self.calc_frequency_modulation()
        if self.pitch is None:
            self.calc_pitch()

        acoustic_features = {"Goodness" : self.goodness, 
                             "Mean_frequency" : self.mean_frequency, 
                             "Entropy" : self.entropy, 
                             "Amplitude" : self.amplitude, 
                             "Amplitude_modulation" : self.AM, 
                             "Frequency_modulation" : self.FM, 
                             "Pitch" : self.pitch}
        return acoustic_features
    
    def calc_feature_stats(self):
        """Calculate summary statistics for acoustic features over a song interval.
        This method returns a dataframe containing the mean, min, max, std, 25th percentile, 
        50th percentile and 75th percentile values for all acoustic features (Goodness, 
        Mean_frequency, Amplitude, Amplitude_modulation, Frequency_modulation and Pitch) across 
        the current song interval.

        Returns:
            pd.DataFrame: Dataframe with a column for each acoustic feature, with the different 
            summary statistics in each row.
        """
        #calculate all acoustic features
        acoustic_features = self.calc_all_features()
        #convert acoustic features dict to dataframe
        acoustic_features = pd.DataFrame(acoustic_features)
        #calculate summary statistics
        acoustic_features_summary = acoustic_features.describe().drop('count') #count is just the number of windows which isn't important. 
        return acoustic_features_summary
    
    def save_features(self, out_file_path, file_name):
        """Save acoustic features and metadata as .csv files

        Saves a table with the acoustic features for each window in the 
        song interval as a .csv file called  `file_name_features.csv`. 
        It also saves a .csv file called `file_name_metadata.csv` with 
        all the hyperparameter values used to calculate the features, 
        as well as the avn version, the original file name and the 
        onset and offset timestamps.  

        Args:
            out_file_path (str): Path to a folder in which to save the .csv files. Must end in '/'. 
            file_name (str): name of the file to serve as the root name for the `_features.csv` and 
            `_metadata.csv` files. 
        """

        #Get acoustic features
        acoustic_features = self.calc_all_features()
        #convert to dataframe
        acoustic_features = pd.DataFrame(acoustic_features)
        #get hyperparameters and convert them to dataframe
        hyperparams = pd.DataFrame(self._get_hyperparameters())

        #save acoustic features
        acoustic_features.to_csv(out_file_path + file_name + "_features.csv")
        #save hyperparameters
        hyperparams.to_csv(out_file_path + file_name + "_metadata.csv")

    def save_feature_stats(self, out_file_path, file_name):
        """Save summary statistics for acoustic features and metadata as .csv files
        Saves a table with summary statistics (mean, max, min, std, 25th, 50th and
        75th percentiles) for each acoustic feature across the song interval as a 
        .csv file called `file_name_feature_stats.csv`.
        It also saves a .csv file called `file_name_metadata.csv` with 
        all the hyperparameter values used to calculate the features, 
        as well as the avn version, the original file name and the 
        onset and offset timestamps.

        Args:
            out_file_path (str): Path to a folder in which to save the .csv files. Must end in '/'. 
            file_name (str): name of the file to serve as the root name for the `_feature_stats.csv` 
            and `_metadata.csv` files. 
        """
        #get acoustic feature summary
        feature_stats = self.calc_feature_stats()
        #get hyperparmeters/ metadata and convert to dataframe
        hyperparams = pd.DataFrame(self._get_hyperparameters())

        #save feature stats
        feature_stats.to_csv(out_file_path + file_name + "_feature_stats.csv")
        #save hyperparameters
        hyperparams.to_csv(out_file_path + file_name + "_metadata.csv")

    def _get_hyperparameters(self):
        """Get all hyperparameters and metadata in dict
        Assembles a dictionary containing all the necessary information 
        to reproduce the current feature calculations. This is saved
        as a .csv file automatically when the acoustic feature or acoustic
        feature summary statistics are saved using the `.save_features()` 
        or `.save_feature_stats()` methods

        Returns:
            dict: dictionary containing all the necessary information to 
            reproduce the current feature calculations. 
        """
        #create dictionary of hyperparameters and metadata
        hyperparams = {'Date' : [datetime.date.today().strftime('%Y-%m-%d')],
                       'file' : [self.full_file.file_name], 
                       'onset' : self.onset, 
                       'offset' : self.offset,
                       'avn_version' : [avn.__version__], 
                       'win_length' : self.win_length, 
                       'hop_length' : self.hop_length, 
                       'n_fft' : self.n_fft, 
                       'max_F0' : self.max_F0, 
                       'min_frequency' : self.min_frequency, 
                       'freq_range' : self.freq_range, 
                       'baseline_amp' : self.baseline_amp, 
                       'fmax_yin' : self.fmax_yin}
        return hyperparams


