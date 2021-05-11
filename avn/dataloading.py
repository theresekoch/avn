# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 09:05:15 2021

@author: Therese
"""

from scipy.io import wavfile
import librosa
import re
import scipy.signal
import pandas as pd


class SongFile:
    """
    Data and metadata pertaining to a single audio file.
    
    
    Attributes
    ----------
    data: ndarray
        Contains audio data of wavfile. 
    
    sample_rate: int
        Sample rate of song data. Based on native sample rate of wavfile.
    
    duration: float
        Duration of the audio file in seconds. 
    
    file_path: str
        Path to the local .wav file used to instantiate the SongFile object.
    
    Methods
    -------
    bandpass_filter(lower_cutoff, upper_cutoff)
        Applies a hamming window bandpass filter to the audio data. 
    
    """
    def __init__(self, file_path):
        """
        Parameters
        ----------
        file_path : str
            Path to the local .wav file to be loaded as a SongFile object.
             
        """
        self.sample_rate, self.data = wavfile.read(file_path)
        self.data = self.data.astype(float)
        self.duration = librosa.get_duration(self.data, sr = self.sample_rate)
        self.file_path = file_path
        
        #get file name -- This may be windows specific. 
        file_name_regex = re.compile("\\\\")
        self.file_name = file_name_regex.split(self.file_path)[-1]
                                                  
    def bandpass_filter(self, lower_cutoff, upper_cutoff):
        """
        Applies a hamming window bandpass filter to the audio data.

        Parameters
        ----------
        lower_cutoff : int
            Lower cutoff frequency in Hz for the filter. 
        upper_cutoff : int
            Upper cutoff frequency in Hz for the filter. 

        Returns
        -------
        None.

        """
        #create hamming window filter
        filter_bandpass = scipy.signal.firwin(101, cutoff = [lower_cutoff, upper_cutoff], 
                                              fs = self.sample_rate, 
                                              pass_zero = False)
        #apply filter to audio data
        self.data = scipy.signal.lfilter(filter_bandpass, [1.0], self.data)
        
        

class Utils:
    """
    Contains data loading utilities.
    
    """
    def __init__(self):
        """
        Initialize a Utils class for data loading. 
        """
        pass
    
    def clean_seg_table(syll_table): 
        """
        Reformats syllable data frames imported from evsonganaly, so that they 
        have the same format as avn generated seg_tables. 
        
        Parameters
        ----------
        syll_table : pandas DataFrame
            Dataframe imported from a csv containing syllable segmentation and
            labeling generated in evsonganaly in MATLAB. 

        Returns
        -------
        syll_table : pandas DataFrame
            seg_table style dataframe, with syllable onset and offset times, 
            labels  and file names. 
        """

        #convert timestamps from miliseconds to seconds
        syll_table['onsets'] = syll_table['onsets'] / 1000
        syll_table['offsets'] = syll_table['offsets'] / 1000
        
        #remove .not.mat file extension from file names in files column
        syll_table['files'] = syll_table['files'].str.split('.not', 1).str[0]
        
        return syll_table
      
    def add_ev_song_truth_table(seg_data, file_path):
        """
        Loads a 'ground truth' segmentation file generated in evsonganaly, and 
        adds it as a `.true_seg_table` attribute to the provided `seg_data` object. 

        Parameters
        ----------
        seg_data : avn.segmentation.SegData object
            SegData object containing segmentations of files corresponding to 
            the evsonganaly ground truth segmentation in the file indicated by 
            `file_path`
        file_path : str
            String containing the full file path to a 'ground truth' segmentation
            .csv file generated with evsonganaly. 

        Returns
        -------
        seg_data : avn.segmentation.SegData object
            Same `seg_data` object as passed as input, but with the added 
            `.true_seg_table` attribute, containing segmentation information from 
            the file indicated by `file_path`

        """
        #load ground truth segmentation table from csv
        true_seg_table = pd.read_csv(file_path)
        
        #conver timestamps from miliseconds to seconds
        true_seg_table['onsets'] = true_seg_table['onsets'] / 1000
        true_seg_table['offsets'] = true_seg_table['offsets'] / 1000
        
        #remove .not.mat file extension from file names in file column
        true_seg_table['files'] = true_seg_table['files'].str.split('.not', 1).str[0]
        
        #add reformated ground truth segmentation table as an attribute of seg_data
        seg_data.true_seg_table = true_seg_table
        
        return seg_data