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
    
    def __init__(self, file_path):
        self.sample_rate, self.data = wavfile.read(file_path)
        self.data = self.data.astype(float)
        self.duration = librosa.get_duration(self.data, sr = self.sample_rate)
        self.file_path = file_path
        
        #get file name -- This may be windows specific. 
        file_name_regex = re.compile("\\\\")
        self.file_name = file_name_regex.split(self.file_path)[-1]
                                                  
    def bandpass_filter(self, lower_cutoff, upper_cutoff):
        #create filter
        filter_bandpass = scipy.signal.firwin(101, cutoff = [lower_cutoff, upper_cutoff], 
                                              fs = self.sample_rate, 
                                              pass_zero = False)
        
        self.data = scipy.signal.lfilter(filter_bandpass, [1.0], self.data)
        
        
        
#    def load_wav(self, file_path):
 #       self.sample_rate, self.data = wavfile.read(file_path)
  #      self.data = self.data.astype(float)
   #     self.file_duration = librosa.get_duration(self.data, sr = self.sample_rate)
        
    
#curr_file = song_file()

#curr_file.load_wav("E:/Grad_School/Code_and_software/Py_code/March_2021_redo/redo_data/segmented_songs/B145/B145_42278.29782241_10_1_8_16_22.wav")

class Utils:
    
    def __init__(self):
        pass
    
    def clean_seg_table(syll_table): 
          '''
          Reformats syll tables imported from evsonganaly so that they are compatible
          with python generated ones
        
          Inputs
          ----
          syll_table: Pandas Dataframe, imported from a csv containing evsonganaly 
          segmentation and labeling info
        
          Outputs
          ----
          syll_table: Pandas Dataframe, now with corrected file names and timestamps in seconds
        
          Notes
          -----
          This function specifically removes .not.mat file extensions from the file names
          so that they are simply .wav and can be compared to file names in the 
          segmentation generated syllable tables. It also converts the timestamps of 
          onsets and offsets from miliseconds to seconds, again so that it is consistent
          with the segmentation resuls
          ''' 
        
          syll_table['onsets'] = syll_table['onsets'] / 1000
          syll_table['offsets'] = syll_table['offsets'] / 1000
        
          syll_table['files'] = syll_table['files'].str.split('.not', 1).str[0]
        
          return syll_table
      
    def add_ev_song_truth_table(segData, file_path):
        true_seg_table = pd.read_csv(file_path)
        
        true_seg_table['onsets'] = true_seg_table['onsets'] / 1000
        true_seg_table['offsets'] = true_seg_table['offsets'] / 1000
        
        true_seg_table['files'] = true_seg_table['files'].str.split('.not', 1).str[0]
        
        segData.true_seg_table = true_seg_table
        
        return segData