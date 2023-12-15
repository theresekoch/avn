# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:30:22 2023

@author: Therese
"""
import avn.dataloading as dataloading
import avn.acoustics as acoustics
import avn.syntax as syntax
import librosa
from scipy.signal.windows import hann
import scipy.signal
from scipy.stats import entropy
import numpy as np
import pandas as pd
from numpy.fft import fft, fftfreq
import glob
import matplotlib.pyplot as plt
import seaborn as sns



class RhythmAnalysis:

    def __init__(self, Bird_ID):

        self.Bird_ID = Bird_ID

    def make_rhythm_spectrogram(self, song_files = None, 
                                song_folder_path= None,
                                frame_length=3, 
                                derivative=True,
                                padded_length = 100000,
                                max_frequency = 30, 
                                hop_length = 0.2,
                                n_windows = 3):
        """make rhythm spectrogram dataframe

        Generate a rhythm spectrogram using all wav files is song_folder_path, or 
        all files inn song_file. This rhythm spectrogram will have one column per file,
        where each column represents the rhythm spectrum of that file. 
        
        Files containing typical, mature zebra finch song will have prominent harmonic banding 
        in their rhythm spectra, which will be consistent from file to file, resulting
        in a rhythm spectrogram with horizontal stripes and little jitter. 

        Immature birds, or birds with very inconsistent song timing will have power more
        evenly spread across the frequency bands in their rhythm spectrogram, and/or
        be less consistent from file to file. 

        For more detailed information on how the rhythm spectrum is calculated for each file, 
        see `RhythmAnalysis.rhythm_spectrum_single_file()`. 

        :param song_files: A value must be provided for `song_files` OR `song_folder_path`, but not both. 
            List containing the paths to .wav files to include in the rhythm spectrogram.
        :type song_files: list of strings, optional
        :param song_folder_path: A value must be provided for `song_files` OR `song_folder_path`, but not both. 
            Path to a folder containing .wav files to include in the rhythm spectrogram. All .wav files of sufficient 
            duration will be included from this folder. 
        :type song_folder_path: str, optional
        :param frame_length: length in seconds of each frame within the .wav file over which to compute a spectrum, defaults to 3
        :type frame_length: int, optional
        :param derivative: If True, the rhythm spectrum will be calculated on the first derivative of the amplitude. 
            If False, the rhythm spectrum will be calculated directly on this amplitude. This generally results in all 
            energy being concentrated at very low frequencies, making it difficult to see harmonic structure. defaults to True
        :type derivative: bool, optional
        :param padded_length: size in frames to which to pad the amplitude before calculating its spectrum. 
            Larger values will result in smoother interpolation of points in the spectrum. defaults to 100000
        :type padded_length: int, optional
        :param max_frequency: maximum frequency in Hz to include in the rhythm spectrum, defaults to 30
        :type max_frequency: int, optional
        :param hop_length: time between windows in seconds when windowing .wav file, defaults to 0.2
        :type hop_length: float, optional
        :param n_windows: number of windows over which to average the rhythm spectra when calculating the
            rhythm spectrum of a single file, defaults to 3
        :type n_windows: int, optional
        :return: rhythm spectrum 
        :rtype: pd.DataFrame()
        """
        
        #save parameters as attributes for future reference and export
        self.frame_length = frame_length
        self.derivative = derivative
        self.padded_length = padded_length
        self.max_frequency = max_frequency
        self.hop_length = hop_length
        self.n_windows = n_windows

        #if both song_folder_path and song_files are provided, raise an error because the files to use are ambiguous.
        if (song_folder_path != None) and (song_files != None):
            raise RuntimeError("You must specify either song_files OR song_folder_path, not both. As both were specified, the files to use for the rhythm spectrogram are ambiguous.")
        
        #if list of song files isn't provided, get list of all song files in folder
        if song_files is None:
            #if the folder also isn't provided, we can't make the rhythm spectrum
            if song_folder_path is None:
                raise RuntimeError("You must specify either song_files or song_folder_path. As neither were specified the rhythm spectrogram cannot be made.")
            else:
                #get list of song files in song_folder_path
                song_files = glob.glob(song_folder_path + "*.wav")
        
        

        #initialize empty dataframe to store rhythm spectrogram
        full_rhythm_spectrogram = pd.DataFrame()

        #Calculate rhythm spectrum for each file and concatenate them to make a rhythm spectrogram
        for song_file_path in song_files:

            rhythm_spectrum = self.rhythm_spectrum_single_file(song_file_path, 
                                                            frame_length=frame_length, 
                                                            derivative=derivative,
                                                            padded_length = padded_length,
                                                            max_frequency = max_frequency, 
                                                            hop_length = hop_length,
                                                            n_windows = n_windows)
            #check that rhythm_spectrum_single_file didn't return None. 
            #this happens if a file is shorter than frame_length. We will simply skip those files. 
            if not rhythm_spectrum is None:
                full_rhythm_spectrogram = pd.concat([full_rhythm_spectrogram, rhythm_spectrum], axis = 1)
        
        self.rhythm_spectrogram = full_rhythm_spectrogram

        return full_rhythm_spectrogram

    def rhythm_spectrum_single_file(self,
                                    song_file_path, 
                                    frame_length=3, 
                                    derivative=True,
                                    padded_length = 100000,
                                    max_frequency = 30, 
                                    hop_length = 0.2,
                                    n_windows = 3):
        """Calculate rhythm spectrum of a single .wav file

        This calculates the rhythm spectrum of a single song file. 
        Birds with typical mature song will have prominent harmonic banding
        in their rhythm spectra, whereas immature or otherwise variable birds
        will have energy more evenly spread across frequency bands. 

        The rhythm spectrum of a signal is a fourier transform of the 
        derivative of the amplitude of that signal, however this doing this over 
        the full .wav file won't represent rhythms present within only smaller 
        portions of the file. To overcome this, this function actually breaks 
        the audio file into multiple windows, each `frame_length` in duration, 
        separated by an offset of `hop_length`. From among the resulting windows, 
        the top `n_windows` with the highest amplitude will be selected, as these
        likely contain the most song. The rhythm spectrum will be calculated for 
        each of those windows, then averaged across them to produce a single rhythm
        spectrum representing the song contained in that file. 

        :param song_file_path: path to the .wav file to process. 
        :type song_file_path: str
        :param frame_length: length in seconds of each frame within the .wav file over which to compute a spectrum, defaults to 3
        :type frame_length: int, optional
        :param derivative: If True, the rhythm spectrum will be calculated on the first derivative of the amplitude. 
            If False, the rhythm spectrum will be calculated directly on this amplitude. This generally results in all 
            energy being concentrated at very low frequencies, making it difficult to see harmonic structure. defaults to True
        :type derivative: bool, optional
        :param padded_length: size in frames to which to pad the amplitude before calculating its spectrum. 
            Larger values will result in smoother interpolation of points in the spectrum. defaults to 100000
        :type padded_length: int, optional
        :param max_frequency: maximum frequency in Hz to include in the rhythm spectrum, defaults to 30
        :type max_frequency: int, optional
        :param hop_length: time between windows in seconds when windowing .wav file, defaults to 0.2
        :type hop_length: float, optional
        :param n_windows: number of windows over which to average the rhythm spectra when calculating the
            rhythm spectrum of a single file, defaults to 3
        :type n_windows: int, optional
        :return: mean rhythm spectrum across windows in the .wav file. Index reflects the frequency in Hz.
        :rtype: pd.Series
        """
        #load song from file
        song = dataloading.SongFile(song_file_path)
        #Check if the file is long enough
        if song.duration < (frame_length + n_windows * hop_length):
            print("file is too short for the specified frame_length, n_windows and hop_length. File will be skipped.")
            return None
        
        #split the audio into frames of length frame_length, with a hop length of hop_length
        windowed_signal = librosa.util.frame(song.data, frame_length = int(song.sample_rate * frame_length), hop_length = int(song.sample_rate * hop_length))
        #get the amplitude of each window
        window_amplitudes = [np.sum(np.abs(window)) for window in windowed_signal.T]
        #get the indices of the n_windows windows in the file with the highest total amplitude. These are the windows most likely to contain song. 
        top_n_windows = np.argpartition(window_amplitudes, n_windows)[-n_windows:]

        #initialize empty lists to store the rhythm spectra of each of the top n_windows windows
        rhythm_spectrogram = []
        all_frequency_bands = []

        #loop over each window and calculate it's rhythm spectrum
        for window_arg in top_n_windows:
            #get the audio for the current window
            max_amp_window = windowed_signal[:, window_arg]
            #this is a bit hacky, but overwrite the current Song File's data with the current frame
            song.data = max_amp_window
            song.duration = frame_length
            #create a SongInterval object from the current song, and use that to calculate the amplitude
            #of the window, as SAP does. 
            song = acoustics.SongInterval(song)
            amplitude = song.calc_amplitude()

            #calculate the rhythm spectrum of the amplitude, and the frequency bands 
                #corresponding to each value in the rhythm spectrum
            rhythm_spectrum, frequency_bands = self._rhythm_spectrum_single_window(amplitude=amplitude, 
                                                                            frame_length=frame_length, 
                                                                            derivative=derivative,
                                                                            padded_length=padded_length, 
                                                                            max_frequency=max_frequency)
            #add current window's frequency bands and rhythm spectrum to the list with other windows
            all_frequency_bands.append(frequency_bands)
            rhythm_spectrogram.append(rhythm_spectrum)
        #convert lists of rhythm spectra and frequency bands to a dataframe
        rhythm_spectrogram = np.array(rhythm_spectrogram).T
        rhythm_spectrogram = pd.DataFrame(rhythm_spectrogram, index = frequency_bands).sort_index(ascending=False)
        #get the mean rhythm spectrum across all windows
        mean_rhythm_spectrum = rhythm_spectrogram.mean(axis = 1)
        #normalize amplitudes in spectrum by dividing by the sum
        mean_rhythm_spectrum = mean_rhythm_spectrum / mean_rhythm_spectrum.sum()

        return mean_rhythm_spectrum
    
    def _rhythm_spectrum_single_window(self, amplitude, 
                                       frame_length=3,
                                       derivative=True,
                                       padded_length = 100000,
                                       max_frequency = 30):
        #get sampling frequency of amplitude
        fs = len(amplitude) / frame_length

        #if derivative == True, take the derivative of the amplitude, and calculate the rhythm spectrum of that.
        if derivative:
            amplitude = np.diff(amplitude)
        
        #Center the amplitude at 0 by subtracting the mean
        amplitude = (amplitude - np.mean(amplitude))
        #Apply a hanning window to the amplitude. This will reduce spectral leakage. 
        hann_window = hann(len(amplitude))
        amplitude = amplitude * hann_window

        #Pad the amplitude with zeros to the length `padded_length`. This won't improve the resolution
            #of the FFT, but it will result in more interpolated values, making the spectrum smoother.
        amplitude = self._symmetrical_pad(amplitude, padded_length)
        #apply a bandpass filter to the amplitude, cutting off frequency components below 1hz and above 500hz. 
        filter_bandpass = scipy.signal.firwin(101, cutoff=[1, 500], fs = fs, pass_zero=False)
        amplitude = scipy.signal.lfilter(filter_bandpass, [1.0], amplitude)

        #Calculate the spectrum of the amplitude
        rhythm_spectrum = abs(fft(amplitude))
        #get the frequency corresponding to each value in the rhythm spectrum
        frequency_bands = fftfreq(len(amplitude), d = round(1/fs, 6)) #I have to round otherwise not all files have exactly the same range. I'm not sure why

        #select only the portion of the rhythm spectrum below max_frequency.
        rhythm_spectrum = rhythm_spectrum[(frequency_bands > 0) & (frequency_bands < max_frequency)]
        frequency_bands = frequency_bands[(frequency_bands > 0) & (frequency_bands < max_frequency)]

        return rhythm_spectrum, frequency_bands

    def _symmetrical_pad(self, array, padded_length):
        """symmetrically pad array so it has length padded_length

        Pad the given array with zeros so that it's total length 
        is equal to padded_length. If padded_length - len(array) is 
        odd, it cannot be padded perfectly symmetrically, so one more
        frame will be added on the right than on the left

        :param array: numpy array
        :type array: np.array
        :param padded_length: desired length of array after padding
        :type padded_length: int
        :return: padded array
        :rtype: np.array
        """

        #get frames to pad  on either side of the array
        to_pad = (padded_length - len(array))/2

        #if the to_pad value is not an integer (ie if padded_length - len(array is odd))
        #update the to_pad values so that there is one more frame added to the left of the array
        #and the total length will still == padded_length
        if to_pad % 1:
            pad_right = int(to_pad - 0.5)
            pad_left = int(to_pad + 0.5)
        else:
            pad_left = int(to_pad)
            pad_right = int(to_pad)

        padded_array = np.pad(array, (pad_left, pad_right))
        #check that the legth of the padded array matches the specified padded_length
        assert(len(padded_array) == padded_length)

        return padded_array

    def plot_rhythm_spectrogram(self,
                                figsize = (6, 6),
                                cbar = False, 
                                title = None, 
                                smoothing_window = 1, cmap = 'rocket'):
        """Plot rhythm spectrogram

        Plots the rhythm spectrogram of the given RhythmAnalysis object. This requires that the 
        object already have a .rhythm_spectrogram attribute, which is created by running 
        `rhythm_analysis.make_rhythm_spectrogram()`.

        :param figsize: width and height of figure in inches, defaults to (6, 6)
        :type figsize: tuple, optional
        :param cbar: If True, the colorbar will be including in the figure. If False, it will be omitted. defaults to False
        :type cbar: bool, optional
        :param title: Plot's title. If not specified, the bird ID associated with the `RhythmAnalysis` object will be used as the title.
             defaults to None
        :type title: string, optional
        :param smoothing_window: Size of smoothing window in song files to apply to the rhythm spectrogram. A value of 1 results in no smoothing. 
            Values greater than one result in the mean of `smoothing_window` spectra being displayed. Higher `smoothing_window` 
            values can obscure rendition-to-rendition variability, but make it easier to see consistent harmonic patterns in the 
             rhythm spectrogram, when present. defaults to 1
        :type smoothing_window: int, optional
        :param cmap: matplotlib color map name, defaults to 'rocket'
        :type cmap: str, optional
        :return: figure of the rhythm spectrogram
        :rtype: matplotlib.figure
        """
        if not hasattr(self, 'rhythm_spectrogram'):
            raise RuntimeError("You must first create the rhythm spectrogram by calling the function `.make_rhythm_spectrogram()`.")
        
        rhythm_spectrogram = self.rhythm_spectrogram

        #round index for nicer y axis tick labels
        rhythm_spectrogram.index = [round(x, 2) for x in np.array(rhythm_spectrogram.index)]
        #Number columns 
        rhythm_spectrogram.columns = np.arange(rhythm_spectrogram.shape[1])
        #apply smoothing window to spectrogram
        rhythm_spectrogram = rhythm_spectrogram.rolling(smoothing_window, axis = 1).mean().dropna(axis = 1)
        if title is None:
            title = self.Bird_ID

        #make figure
        fig = plt.figure(figsize = figsize)
        sns.heatmap(rhythm_spectrogram, cbar = cbar, cmap = cmap)
        plt.title(title)
        plt.xlabel('Song File')
        plt.ylabel('Hz')

        return fig

    
    def calc_rhythm_spectrogram_entropy(self):
        """Calculate the Weiner entropy of the mean rhythm spectrum

        Calculate the Weiner entropy of the mean rhythm power spectrum 
        based on the RhythmAnalysis object's `.rhythm_spectrogram` attribute. 
        To create this attribute, please first run `rhythm_analysis.make_rhythm_spectrogram()`
        This is calculated in the same way as the acoustic feature 'entropy'. 
        It is a measure of the uniformity of power spread across frequency bands. 
        The output is log-scaled and can range from 0 to negative infinity. 
        Scores closer to 0 indicate a broader spread of power across bands, 
        and are typical of birds with very variable song timing, such as juvenile birds. 

        :return: The log-scaled Weiner entropy of the mean rhythm power spectrum across all 
            spectra included in `rhythm_spectrogram`.
        :rtype: float
        """
        if not hasattr(self, 'rhythm_spectrogram'):
            raise RuntimeError("You must first create the rhythm spectrogram by calling the function `.make_rhythm_spectrogram()`.")
        
        rhythm_spectrogram = self.rhythm_spectrogram
        #Square the values of the rhythm spectrogram to get the rhythm *power* spectrogram
        power_spectrogram = (rhythm_spectrogram**2)
        #get the average spectrum across all files
        power_spectrum = power_spectrogram.mean(axis = 1)
        #calculate the Weiner entropy of the mean power spectrum
        sum_log = np.sum(np.log(power_spectrum)) 
        log_sum = np.log(np.sum(power_spectrum)/(len(power_spectrum))) 
        entropy = sum_log / (len(power_spectrum)) - log_sum
        return entropy
    
    def get_refined_peak_frequencies(self, freq_range = 3):
        """get frequencies with peak magnitude in rhythm spectrogram.

        In addition to the entropy of the mean spectrum, we can investigate 
        rendition-to-rendition timing variability by looking at the variability
        of the frequency with the highest magnitude in the rhythm spectrogram. 
        To do this reliably, we first find the median of the peak frequency across 
        all frames in the rhythm spectrogram. Then, we restrict the peak frequency 
        search to the `freq_range` frequencies centered on the median. This helps 
        reduce variability caused by the peak jumping to different harmonic bands, 
        as this doesn't reflect a meaningful differences in the timing structure. 
        The resulting peaks can then be plotted with `.plot_peak_frequencies()`
        and their CV can be calculated with `.calc_peak_frequency_cv()`. 

        :param freq_range: range of frequencies in Hz centered on the median peak frequency 
            within which to search for the refined peak frequency. For example, if a bird's median
            peak frequency is 15Hz when we consider the full rhythm spectrum, and the freq_range is 3 Hz, 
            then the refined peak frequency would be the peak frequency between 15-1.5 Hz and 15+1.5 Hz
            across all files in the rhythm spectrogram. This should be set to the largest value possible,
            while still being less than the distance between harmonic bands in a bird with clear harmonic
            structure in their rhythm spectrogram. defaults to 3
        :type freq_range: int, optional
        :return: dataframe containing the refined peak frequency for each file in the rhythm spectrogram.
        :rtype: pd.DataFrame
        """

        if not hasattr(self, 'rhythm_spectrogram'):
            raise RuntimeError("You must first create the rhythm spectrogram by calling the function `.make_rhythm_spectrogram()`.")
        
        self.freq_range = 3
        rhythm_spectrogram = self.rhythm_spectrogram

        #first, get the peak frequencies within the full frequency range of the spectrogram
        peak_frequencies = self._get_peak_frequencies(rhythm_spectrogram)

        #find the median peak frequency
        median_peak = peak_frequencies.peak_frequency.median()
        self.median_peak = median_peak

        #restrict the peak frequency search to the freq_range centered on the median peak
        max_idx = np.max(np.argwhere(rhythm_spectrogram.index.values > (median_peak - freq_range/2)))
        min_idx = np.min(np.argwhere(rhythm_spectrogram.index.values < (median_peak + freq_range/2)))
        self.min_idx = min_idx
        restricted_rhythm_spectrogram = rhythm_spectrogram.iloc[min_idx:max_idx]

        #get the peak frequency within that range
        peak_frequencies = self._get_peak_frequencies(restricted_rhythm_spectrogram)

        self.peak_frequencies = peak_frequencies
        return peak_frequencies
    
    def _get_peak_frequencies(self, rhythm_spectrogram):
        """gets peak frequency in provided rhythm spectrogram

        Finds the frequency with the highest magnitude in the 
        rhythm spectrogram for each frame of the spectrogram. 
        If you plan to use the peak frequency to look at the 
        rendition-to-rendition timing variability, please see 
        `.get_refined_peak_frequencies()`.

        :param rhythm_spectrogram: dataframe containing the 
        rhythm spectra for multiple song files. Generated using 
        `.make_rhythm_spectrum()`.
        :type rhythm_spectrogram: pd.DataFrame
        :return: dataframe containing the peak frequency for each file in the rhythm spectrogram.
        :rtype: pd.DataFrame
        """
        #initialize empty dataframe to store peak frequencies
        peak_freq_df= pd.DataFrame()

        #loop over each spectrum in the spectrogram
        for j in range(rhythm_spectrogram.shape[1]):
            example_spectrum = rhythm_spectrogram.iloc[:, j]
            #find the frequency with the highest magnitude
            peak_frequency = example_spectrum.index[example_spectrum.argmax()]
            #store peak frequency in dataframe
            curr_df = pd.DataFrame({'window_idx' : [j], 
                                    'peak_frequency' : [peak_frequency],
                                    'peak_freq_idx' : [example_spectrum.argmax()]})
            #append to df with peak frequencies of each window.
            peak_freq_df = pd.concat([peak_freq_df, curr_df])
        
        return peak_freq_df
    
    def calc_peak_frequency_cv(self):
        """calculate the cv of the peak frequencies in the rhythm spectrum

        Calculates the coefficient of variation (CV) of the frequency with 
        the highest magnitude in the rhythm spectrum of a bird, as a measure
        of the rendition-to-rendition song timing variability. To calculate 
        this you must first generate the table of peak frequencies using 
        `.get_refined_peak_frequencies()`.

        :return: CV of peak frequency
        :rtype: float
        """

        peak_frequencies = self.peak_frequencies
        #get mean and std of peak frequencies
        mean_freq = peak_frequencies.peak_frequency.mean()
        std_freq = peak_frequencies.peak_frequency.std()

        #calculate CV of peak frequencies
        cv = std_freq/mean_freq

        return cv
    
    def plot_peak_frequencies(self, 
                              figsize = (6, 6),
                              cbar = False, 
                              title = None, 
                              cmap = 'rocket', 
                              s = 15, 
                              color = 'aqua'):
        """plot the peak frequencies over the rhythm spectrogram

        Plots a point over the rhythm spectrogram at the frequency
        band with the highest magnitude in that frame. This can help illustrate 
        rendition-to-rendition variability in the rhythm spectrogram. 
        To plot this you must first generate the table of peak frequencies using 
        `.get_refined_peak_frequencies()`.

        :param figsize: width and height of figure in inches, defaults to (6, 6)
        :type figsize: tuple, optional
        :param cbar: If True, the colorbar will be including in the figure. If False, it will be omitted. defaults to False
        :type cbar: bool, optional
        :param title: Plot's title. If not specified, the bird ID associated with the `RhythmAnalysis` object will be used as the title.
             defaults to None
        :type title: string, optional
        :param cmap: matplotlib color map name, defaults to 'rocket'
        :type cmap: str, optional
        :param s: size of peak frequency marker in points, defaults to 15
        :type s: int, optional
        :param color: color of peak frequency marker, defaults to 'aqua'
        :type color: matplotlib color specification, optional
        :return: figure of the rhythm spectrogram overlaid with points indicating the peak frequencies for each frame.
        :rtype: matplotlib.figure
        """
        peak_frequencies = self.peak_frequencies.copy(deep = True)
        #get plot of peak frequencies
        fig = self.plot_rhythm_spectrogram(figsize=figsize, cbar = cbar, title = title, cmap = 'rocket')
        #overlay peak frequencies
        peak_frequencies.window_idx = peak_frequencies.window_idx + 0.5
        peak_frequencies.peak_freq_idx = peak_frequencies.peak_freq_idx + self.min_idx
        sns.scatterplot(data = peak_frequencies, x = 'window_idx', y = 'peak_freq_idx', s = s, color = color)
        return fig
    

class SegmentTiming:
    
    def __init__(self, Bird_ID, syll_df, song_folder_path):

        self.Bird_ID = Bird_ID
        self.syll_df = syll_df
        self.song_folder_path = song_folder_path

        #make sure there are no negative onset times in syll_df
        #by replacing all negative onsets with an onset time of 0.
        self.syll_df['onsets'] = self.syll_df.onsets.where(syll_df.onsets > 0, 0)

        #some functions borrowed from the syntax module require that syll_df have a 
        #labels column, but the contents don't matter. If no labels column is provided
        #add one where every syllable has the label 's' for syllable. 
        if not 'labels' in syll_df.columns:
            syll_df['labels'] = 's'

    def get_syll_durations(self):
        """Get syllable durations in seconds

        Adds a column to the `SegmentTiming.syll_df` dataframe with the duration of each syllable 
        in seconds.

        :return: copy of `SegmentTiming.syll_df` with a new column, `durations`, containing the 
                duration of each syllable in seconds.
        :rtype: pd.DataFrame
        """
        self.syll_df['durations'] = (self.syll_df.offsets - self.syll_df.onsets)

        return self.syll_df
    
    def get_gap_durations(self, max_gap = 0.2):
        """Get gap durations in seconds

        Create a `gap_df` attribute to your `SegmentTiming` instance with 
        the onset, offset, and duration of all silent gaps in the segmentation
        provided by `SegmentTiming.syll_df`, excluding any gaps longer than
        `max_gap`.

        :param max_gap: maximum gap duration in seconds. This should be set such that
          it is longer than all gaps between syllables in a bout, but is shorter than the 
          gaps between bouts. Defaults to 0.2
        :type max_gap: float, optional
        :return: copy of `SegmentTiming.gap_df`, containing the durations of each gap in seconds
        :rtype: pd.DataFrame
        """
        self.max_gap = max_gap
        #create syntax data object to generate gap df
        syntax_data = syntax.SyntaxData(self.Bird_ID, self.syll_df)
        syntax_data.add_file_bounds(self.song_folder_path)
        self.gap_df = syntax_data.get_gaps_df()
        #filter out any gaps longer than max_gap
        self.gap_df = self.gap_df[self.gap_df.durations < max_gap]
        self.gap_df = self.gap_df.reset_index(drop = True)

        return self.gap_df

    def calc_syll_duration_entropy(self):
        """Calculate entropy of syllable duration distribution

        Calculates the shannon entropy of the bird's syllable 
        duration distribution. Results range from 0 to 1, with 
        higher scores indicating less predictable syllable durations, 
        consistent with the songs of immature birds.
        Based on Goldberg & Fee, 2011.

        :return: shannon entropy of syllable duration distribution
        :rtype: float
        """

        #check if syllable durations have been added yet.
        #if not, add it. 
        if not ('durations' in self.syll_df.columns):
            self.get_syll_durations()

        #define bins over which to calculate syllable density
        bins = np.linspace(-2.5, 0, 50)
        #get density of syllable duration distribution in each bin
        density, __ = np.histogram(np.log10(self.syll_df.durations), density=True, bins = bins)
        #calculate the entropy the distribution based on bin densities. 
        syll_entropy = entropy(density) / np.log(len(bins))

        return syll_entropy

    def calc_gap_duration_entropy(self, max_gap=0.2):
        """Calculates entropy of gap duration distribution

        Calculates the shannon entropy of the bird's gap duration 
        distribution. Results range from 0 to 1, with higher scores
        indicating less predictable gap durations, consistent with 
        the songs of immature birds. Based on Goldberg & Fee, 2011. 

        :param max_gap: maximum gap duration in seconds. This should be set such that
          it is longer than all gaps between syllables in a bout, but is shorter than the 
          gaps between bouts. Defaults to 0.2
        :type max_gap: float, optional
        :return: shannon entropy of the gap duration distribution
        :rtype: float
        """

        #check if gap_df has been created yet.
        #if not, add it. 
        if not hasattr(self, 'gap_df'):
            self.get_gap_durations(max_gap = max_gap)
        #if gap_df exists but was created with a different max_gap
            #parameter, remake it with the current max_gap.    
        elif max_gap!=self.max_gap:
            self.get_gap_durations(max_gap = max_gap)

        #define bins over which to calculate gap duration density
        bins = np.linspace(-2.5, 0, 50)
        #get density of gap duration distribution in each bin
        density, __ = np.histogram(np.log10(self.gap_df.durations), density=True, bins = bins)
        #calculate the entropy the distribution based on bin densities. 
        gap_entropy = entropy(density) / np.log(len(bins))

        return gap_entropy