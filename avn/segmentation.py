# -*- coding: utf-8 -*-
"""
Created on Wed May  5 08:29:00 2021

@author: Therese
"""

import glob
import dataloading
import plotting
import numpy as np
import pandas as pd
import librosa
import librosa.display
import sklearn
#import os
import matplotlib.pyplot as plt
import random

class SegData:
    def __init__(self, Bird_ID, seg_table):
        self.Bird_ID = Bird_ID
        self.seg_table = seg_table
        
    def save_as_csv(self, folder_path):
        #check if SegData has a seg_table attribute. If so, save it. 
        if hasattr(self, 'seg_table'):
            self.seg_table.to_csv(folder_path + self.Bird_ID + "_seg_table.csv")

class Segmenter:
    
    def __init__(self):
        pass
    
    def make_segmentation_table(self, Bird_ID, song_folder_path, upper_threshold, lower_threshold, 
                                max_syll_duration = 0.33):
        
        '''
        Returns DataFrame with the onset and offset timestamp and file name for every syllable
          in every file in the folder corresponding to `Bird_ID`. 
          The output can then be used to calculate 
          the accuracy of segmentation with `get_match_proportions()`, or to crop the 
          spectrogram for syllable clustering. 
        
          Parameters
          ----------
          upper_thresh: int >=0, optional
            This is the threshold used to find syllable onsets. When the MFCC derivative
            surpasses this threshold, the onset of a new syllable is recorded. 
          lower_thresh: int >= 0, optional
            This is the threshold used to find syllable offsets. Normally, the onset of 
            the following syllable will be used as the offset of the previous, so long as
            that doesn't result in a syllable longer than `max_syll_duration`. If that is 
            the case then the the offset is determined by finding the first point after 
            the onset where the RMSE derivative falls below the lower_threshold.
          max_syll_duration: int > 0, optional 
            Maximum possible duration of a single syllable, in seconds. If the gap between
            consecutive syllable onsets is longer than this value, the offset will be 
            determined by lower threshold crossing. 
        
          Returns
          -------
          all_data: DataFrame with columns `onsets`, `offsets`, and `files`, containing 
          the onset and offset timestamps (in seconds) and the file name for every syllable
          in every file in the folder corresponding to the input `Bird_ID`. This can then
          be used to calculate the accuracy of segmentation with 
          `get_match_proportions()`, or to crop the spectrogram for syllable clustering.
        
          Notes
          ------
          This function uses MFCC derivative threshold crossing for syllable segmentation. 
          See also `make_RMSE_segmentation_table()` for an analogous function which 
          uses RMSE threshold crossing for syllable segmentation, and 
          `make_RMSE_segmentation_table` which uses RMSE derivative threshold crossing.  
        
          '''
          
        all_song_files = glob.glob(song_folder_path + "*.wav")
        
        all_data = None
        
        for song_file in all_song_files:
            
            #Load File
            song = dataloading.SongFile(song_file)
        
            #Calculate value of segmentation criteria (e.g. MFCC, RMSE)
            seg_criteria = self.get_seg_criteria(song)
        
            #Create thresholds 
            upper_thresh = self.get_threshold(seg_criteria, thresh = upper_threshold)
            lower_thresh = self.get_threshold(seg_criteria, thresh = lower_threshold)
        
            #Get syllable onset and offset indices
            syll_onset_indices, syll_offset_indices = self.get_syll_onsets_offsets(seg_criteria, 
                                                                              upper_thresh, 
                                                                              lower_thresh, 
                                                                              song.duration,
                                                                              max_syll_duration = max_syll_duration)
            #Convert indices to times -- should this be a function?
            all_times = librosa.frames_to_time(np.arange(len(seg_criteria)), 
                                           sr = song.sample_rate, hop_length = 512)
            syll_onsets = all_times[syll_onset_indices]
            syll_offsets = all_times[syll_offset_indices]
        
            #Package data into Dataframe
            curr_file_data = pd.DataFrame({"onsets": syll_onsets, 
                                           "offsets": syll_offsets})
            curr_file_data['files'] = song.file_name
            
            if all_data is not None:
              all_data = pd.concat([all_data, curr_file_data], ignore_index=True)
            else:
              all_data = curr_file_data
        
        segmentation_data = SegData(Bird_ID, all_data)
        #segmentation_data.seg_table = all_data
        segmentation_data.song_folder_path = song_folder_path
        return segmentation_data
      
    def get_seg_criteria(self, song):
        raise NotImplementedError
    
    def rescale(self, seg_criteria):
        scaler = sklearn.preprocessing.MinMaxScaler()
        seg_criteria = scaler.fit_transform(seg_criteria[:, np.newaxis])[:, 0]
        return seg_criteria
        
    def get_threshold(self, seg_criteria, thresh):
     """
      Returns a flat threshold for comparison to mfcc_derivative
    
      Inputs
      ------
    
      mfcc_derivative: numpy array, 1D
      derivative of the mfcc which is going to be compared to 
      this threshold for syllable segmentation.
    
      thresh: int, optional
      The value of the flat threshold to be created
    
      Returns
      -------
    
      threshold_vector: numpy array 
    numpy array with the same dimensions as `mfcc_derivative`,
    with the value `thresh` at every point. 
    
      Notes
      -----
    
      This function creates a flat threshold. 
      Those are suitable for comparison to a
      derivative, but not to a raw trace (like MFCC or RMSE etc.). 
      In those cases you would likely want an adaptive threshold.
      """
    
     threshold_vector = np.ones_like(seg_criteria) * thresh
    
     return threshold_vector
  
    def get_syll_onsets_offsets(self, seg_criteria, upper_thresh, lower_thresh, total_file_duration, 
                            max_syll_duration = 0.3):
      """
      Returns the onsets and offsets of all syllables in a file based on segmentation
      criteria threshold crossing. 
    
      Inputs
      ------
      seg_criteria: numpy array, 1D
          Array containing value of the segmentation criteria for every point in a given file. 
      upper_thresh: numpy array, 1D
          Array with the same dimensions as `seg_criteria`, which contains the value of 
          the upper threshold for every point in a given file. 
      lower_thresh: numpy array, 1D
          Array with the same dimensions as `seg_criteria` which contains the value of the 
          lower threshold for every point in a given file. 
      total_file_duration: int > 0
          Duration in seconds of the original file, used to convert between frames 
          and seconds. 
      max_syll_duration: int > 0, optional
          Maximum allowable duration (in seconds) for a single syllable. If the 
          gap between consecutive syllable onsets is longer than this value, the 
          offset will be determined by lower threshold crossing. 
    
      Returns
      ----
      syll_onset: numpy array, 1D
          Vector containing the indices of the seg_criteria where there are syllable 
          onsets. 
      syll_offset: numpy array, 1D
          Vector containing the indices of the seg_criteria array where there are syllable
          offsets. 
    
      Notes
      -----
      This segmentation works by first finding every point where the segmentiation 
      criteria (e.g. RMSE, MFCC derivative etc.) crosses the upper threshold. 
      Those are all considered syllable onset points. 
      The onset of the following syllable will be used as the offset of the previous
      syllable, so long as that doesn't result in a syllable with a duration longer
      than `max_syll_duration`. If the syllable is longer, then the offset is 
      determined by finding the first point after the onset where the segmentation
      criteria crosses the lower threshold. If that still results in a syllable 
      which is longer than `max_syll_duration` then the offset defaults to 
      `max_syll_duration`seconds after the syllable onset. 
      """
    
      #get array of boolean above or below upper thresh
      upper_thresh_position = seg_criteria >= upper_thresh
      #get array of positive or negative threshold crossings
      upper_thresh_crossing = np.diff(upper_thresh_position, prepend = 0)
      #get array with True at every upward crossing of upper thresh
      upper_thresh_passing = upper_thresh_crossing == 1
      
      #get array of boolean above or below lower thresh
      lower_thresh_position = seg_criteria < lower_thresh
      #get array of positive or negative threshold crossings
      lower_thresh_crossing = np.diff(lower_thresh_position, prepend = 0)
      #get array with True at every downward crossing of the lower thresh
      lower_thresh_passing = lower_thresh_crossing == 1 
      
      #Get indices of upward crossings of upper thresh
      upper_thresh_indices = np.nonzero(upper_thresh_passing)[0]
      
      #Add end of file to act as a final syllable onset. 
      #This way, all syllables will have an offset at or before the end of the file. 
      upper_thresh_indices = np.append(upper_thresh_indices, len(upper_thresh_passing)-1)
      
      #initialize vector to contain syllable offsets
      syll_offset = np.zeros_like(upper_thresh_indices)
    
      #determine how many frames corresponds to a syllable with duration `max_syll_duration`. 
      max_syll_duration_frames = seg_criteria.shape[0] / total_file_duration * max_syll_duration
      
      #loop through positive upper threshold crossings
      for i in np.arange(len(upper_thresh_indices)-1):
    
      #check whether gap between consecutive offsets is more than max_syll_duration seconds long
        if (upper_thresh_indices[i+1] - upper_thresh_indices[i]) >  max_syll_duration_frames:
          
          #if so, find the first negative lower threshold crossing after positive upper threshold crossing
          #and set that to the syllable offset. 
          for a in np.arange(upper_thresh_indices[i], np.floor(upper_thresh_indices[i] + max_syll_duration_frames)).astype(int):
        
            if lower_thresh_passing[a] == 1:
              syll_offset[i] = a 
              break
          
            #If the lower threshold isn't crossed before `max_syll_duration`s after the onset, 
            #then just set the offset to `max_syll_duration` seconds after onset.
            syll_offset[i] = np.floor(upper_thresh_indices[i] + max_syll_duration_frames)
    
      #Otherwise, the syllable offset is the onset of the following syllable. 
        else: 
          syll_offset[i] = upper_thresh_indices[i + 1]
        
      #Drop the last one because it is an artifact
      syll_offset = syll_offset[:-1]
      syll_onset = upper_thresh_indices[:-1]
    
      return(syll_onset, syll_offset)
      
      
      
class RMSE(Segmenter):
    
    def __init__(self):
        super().__init__()
        
    def get_seg_criteria(self, song, hop_length = 256, n_fft = 2048, 
                         bandpass = True, lower_cutoff = 200, upper_cutoff = 9000, 
                         rescale = True):
        '''
          returns RMSE values for the input song
        
          Parameters
          -------------
          song: instance of avn.dataloading.SongFile class. 
                contains data from a single file of song. 
          hop_length: int > 0, optional
                The number of samples beterrn successive frames used in the short term 
                fourier transform to gerenate RMSE values. 
          n_fft: int > 0, optional
                The length of the FFT window used to calculate the RMSE values
          bandpass: Bool, optioinal
                If True, the wave will be bandpass filtered before calculating the RMSE. 
                If False, the RMSE will be calculated on the wave as-is. 
          lower_cutoff: int >= 0, optional 
              The lower frequency limit used to bandpass filter the input wave 
              before calculating the RMSE features. 
          upper_cutoff: int > lower_cutoff, optional 
              The upper frequency limit used to bandpass filter the input wave
              before calculating the RMSE features. 
          rescale: Bool, optional 
                If True, the RMSE will be rescaled so that all values fall between 0 and 1. 
                This is meant to ensure consistency across recordings. 
                If False, the raw RMSE values will be returned.
          
          
          Output:
          -------
          rmse: numpy array, 1D
              A one dimensional numpy array containing the RMSE values for the input wave. 
          '''
        #apply bandpass filter to wave if `bandpass` = True
        if bandpass:
            song.bandpass_filter(lower_cutoff, upper_cutoff)
        
        #calculate RMSE
        rmse = librosa.feature.rms(y = song.data, 
                                  hop_length = hop_length, 
                                  frame_length = n_fft)[0, :]
                                  
        #rescale all rmse values to a range of 0 to 1 if `rescale` = True
        if rescale:
            rmse = self.rescale(rmse)
        
        return rmse
    
class RMSEDerivative(Segmenter):
    
    def __init__(self):
        super().__init__()
        
    def get_seg_criteria(self, song, hop_length = 256, n_fft = 2048, 
                         bandpass = True, lower_cutoff = 200, upper_cutoff = 9000, 
                         rescale = True, deriv_width = 3):
        '''
          returns first derivative of the RMSE for the input song
        
          Parameters
          -------------
          song: instance of avn.dataloading.SongFile class. 
                contains data from a single file of song. 
          hop_length: int > 0, optional
                The number of samples beterrn successive frames used in the short term 
                fourier transform to gerenate RMSE values. 
          n_fft: int > 0, optional
                The length of the FFT window used to calculate the RMSE values
          bandpass: Bool, optioinal
                If True, the wave will be bandpass filtered before calculating the RMSE. 
                If False, the RMSE will be calculated on the wave as-is. 
          lower_cutoff: int >= 0, optional 
              The lower frequency limit used to bandpass filter the input wave 
              before calculating the RMSE features. 
          upper_cutoff: int > lower_cutoff, optional 
              The upper frequency limit used to bandpass filter the input wave
              before calculating the RMSE features. 
          rescale: Bool, optional 
                If True, the RMSE will be rescaled so that all values fall between 0 and 1
                before the derivative is claculated. This is meant to ensure consistency across recordings. 
                If False, the RMSE derivative will be calculated on the raw RMSE values. 
          
          Output:
          -------
          rmse_derivative: numpy array, 1D
              A one dimensional numpy array containing the first derivative of the 
              RMSE values for the input song.
         '''
        #apply bandpass filter to song if `bandpass` = True
        if bandpass:
            song.bandpass_filter(lower_cutoff, upper_cutoff)
        
        #calculate RMSE
        rmse = librosa.feature.rms(y = song.data,
                                   hop_length = hop_length, 
                                   frame_length = n_fft)[0, :]
        
        #rescale  all rmse values to a range of 0 to 1 if `rescale` = True
        if rescale:
            rmse = self.rescale(rmse)
            
        #calculate derivative of rmse
        rmse_derivative = librosa.feature.delta(rmse, width = deriv_width)
        
        return rmse_derivative
    
    
class MFCC(Segmenter):
    def __init__(self):
        super().__init__()
            
    def get_seg_criteria(self, song, hop_length = 512, n_fft = 2048, 
                         bandpass = True, lower_cutoff = 200, upper_cutoff = 9000, 
                         rescale = True):
        '''
          returns MFCC values for the input song
        
          Parameters
          -------------
          song: instance of avn.dataloading.SongFile class. 
                contains data from a single file of song. 
          hop_length: int > 0, optional
                The number of samples beterrn successive frames used in the short term 
                fourier transform to gerenate MFCC values. 
          n_fft: int > 0, optional
                The length of the FFT window used to calculate the MFCC values
          bandpass: Bool, optioinal
                If True, the wave will be bandpass filtered before calculating the MFCC. 
                If False, the MFCC will be calculated on the wave as-is. 
          lower_cutoff: int >= 0, optional 
              The lower frequency limit used to bandpass filter the input wave 
              before calculating the MFCC features. 
          upper_cutoff: int > lower_cutoff, optional 
              The upper frequency limit used to bandpass filter the input wave
              before calculating the MFCC features. 
          rescale: Bool, optional 
                If True, the MFCC will be rescaled so that all values fall between 0 and 1. 
                This is meant to ensure consistency across recordings. 
                If False, the raw MFCC values will be returned.
          
          
          Output:
          -------
          mfcc: numpy array, 1D
              A one dimensional numpy array containing the MFCC values for the input wave. 
          '''
        #apply bandpass filter to wave if `bandpass` = True
        if bandpass:
            song.bandpass_filter(lower_cutoff, upper_cutoff)
        
        #calculate first MFCC
        mfcc = librosa.feature.mfcc(song.data, song.sample_rate, n_mfcc = 1,
                                    hop_length = hop_length, 
                                    win_length = n_fft)[0, :]
                                  
        #rescale all mfcc values to a range of 0 to 1 if `rescale` = True
        if rescale:
            mfcc = self.rescale(mfcc)
        
        return mfcc
    
class MFCCDerivative(Segmenter):
    def __init__(self):
        super().__init__()
        
    def get_seg_criteria(self, song, hop_length = 512, n_fft = 2048, 
                         bandpass = True, lower_cutoff = 200, upper_cutoff = 9000, 
                         rescale = True, deriv_width = 3):
        '''
          returns first derivative of the first MFCC for the input song
        
          Parameters
          -------------
          song: instance of avn.dataloading.SongFile class. 
                contains data from a single file of song. 
          hop_length: int > 0, optional
                The number of samples beterrn successive frames used in the short term 
                fourier transform to gerenate MFCC values. 
          n_fft: int > 0, optional
                The length of the FFT window used to calculate the MFCC values
          bandpass: Bool, optioinal
                If True, the wave will be bandpass filtered before calculating the MFCC. 
                If False, the MFCC will be calculated on the wave as-is. 
          lower_cutoff: int >= 0, optional 
              The lower frequency limit used to bandpass filter the input wave 
              before calculating the MFCC features. 
          upper_cutoff: int > lower_cutoff, optional 
              The upper frequency limit used to bandpass filter the input wave
              before calculating the MFCC features. 
          rescale: Bool, optional 
                If True, the MFCC will be rescaled so that all values fall 
                between 0 and 1 before the derivative is calculated. 
                This is meant to ensure consistency across recordings. 
                If False, the raw MFCC values will be used.
         deriv_width: int.............. fill this in by looking at librosa documentation. 
          
          
          Output:
          -------
          mfcc_derivative: numpy array, 1D
              A one dimensional numpy array containing the first derivative of 
              the first MFCC of the input song. 
          '''
        #apply bandpass filter to wav if `bandpass` == True
        if bandpass:
            song.bandpass_filter(lower_cutoff, upper_cutoff)
        
        #calculate first MFCC
        mfcc = librosa.feature.mfcc(song.data, song.sample_rate, n_mfcc = 1, 
                                    hop_length = hop_length, 
                                    win_length = n_fft)[0, :]
        
        #rescale MFCC vector is `rescale` == True
        if rescale:
            mfcc = self.rescale(mfcc)
            
        #calculate derivative of rescaled MFCC
        mfcc_derivative = librosa.feature.delta(mfcc, width = deriv_width)
        
        return mfcc_derivative
    

class Metrics():
    
    def __init__(self):
        pass
    
    def calc_F1 (seg_data, max_gap = 0.05):
        '''
        Given a avn.segmentation.Segmenter class object with a seg_table attribute 
        and a table containing ground truth segmentations of the the set of files, 
        this function will calculate the overall F1 score of segmentation
        across all the files. 
        
        '''
        truth_seg_table = seg_data.true_seg_table
        
        all_true_positives = 0
        all_false_positives = 0
        all_false_negatives = 0
        
        #loop through each individual file in the segmenter.seg_table
        for current_file in np.unique(seg_data.seg_table['files']):
            #filter the seg_tables so that they contain only syllables from the 
            #current file. 
            seg_current_file = seg_data.seg_table[seg_data.seg_table['files'] == current_file]
            truth_current_file = truth_seg_table[truth_seg_table['files'] == current_file]
            
            #get best matches of truth to segmenter onsets
            seg_matches = Metrics.get_best_matches(seg_current_file['onsets'], 
                                                truth_current_file['onsets'], max_gap)
            #get best matches of segmenter to truth onsets
            truth_matches = Metrics.get_best_matches(truth_current_file['onsets'], 
                                             seg_current_file['onsets'], max_gap)
            
            #calculate F1 components for the given file
            true_positives, false_positives, false_negatives = Metrics.calc_F1_components_per_file(seg_current_file, 
                                                                                           truth_current_file, 
                                                                                           seg_matches, 
                                                                                           truth_matches, 
                                                                                           max_gap)
            #add F1 components to cumulative sum
            all_true_positives += true_positives
            all_false_positives += false_positives
            all_false_negatives += false_negatives
            
        #calculate F1 score
        seg_data.F1 = all_true_positives / (all_true_positives + 0.5 * (all_false_positives + all_false_negatives))
        seg_data.precision = all_true_positives / (all_true_positives + all_false_positives)
        seg_data.recall = all_true_positives / (all_true_positives + all_false_negatives)
        
        return seg_data
        
        
            
        
    def get_best_matches(first_onsets, second_onsets, max_gap):
        '''
          Finds the best unique matches between timestamps in two sets, calcuated in different
          ways on the same file. These can reflect syllable onset or offset timestamps, 
          although I refer to them only as onsets for simplicity. 
        
          Inputs
          ------
          first_onsets: Pandas Series, Contains the timestamps in seconds of syllable onsets
          calculated with a particular method. 
        
          second_onsets: Pandas Seris, Contains the timestamps in seconds of syllable onsets
          calculated with a different method. 
        
          max_gap: int > 0, optional, maximum allowable time difference  in seconds 
          between onsets where they will still be considered a match.
        
          Outputs
          -------
          best_matches: numpy array, 1D, For every syllable onset in `first_onsets` it 
          contains the index of the best unique match in `second_onsets`. If there is 
          no unique match within the allowable `max_gap`, the value of the match is 
          `NaN`. 
          '''
        first_grid, second_grid = np.meshgrid(first_onsets, second_onsets)
        delta_t_grid = abs(first_grid - second_grid)
        
        #set max gap threshold
        delta_t_grid = np.where(delta_t_grid > max_gap, np.inf, delta_t_grid)
        
        if delta_t_grid.shape[0] == 0:
            best_matches = np.array([])
            return best_matches 
        
        #find best matches
        best_matches = np.argmin(delta_t_grid, axis = 0).astype(float)
        
        #remove matches were delta t is > max_gap
        for i, match_index in enumerate(best_matches):
            if np.isinf(delta_t_grid[int(match_index), i]):
              best_matches[i] = np.NaN 
        
        best_matches_previous = best_matches.copy()   
        
        #Deal with duplicate values by setting the second best matches to their second best pairs
        best_matches, delta_t_grid = Metrics.correct_duplicates(best_matches, delta_t_grid)
        
        #check if there were changes made by checking for duplicates. If so, repeat duplicate check.
        if not np.allclose(best_matches, best_matches_previous):
            best_matches, delta_t_grid = Metrics.correct_duplicates(best_matches, delta_t_grid)
        
        #make sure duplicate corrections didn't result in out of order matches. 
        for i, curr_match in enumerate(best_matches):
             if i+1 < len(best_matches):
               if curr_match > best_matches[i+1]:
                 best_matches[i+1] = np.nan
        
        return best_matches
    
    
    def correct_duplicates(best_matches, delta_t_grid):   
         '''
          Finds any duplicate matches in the set of best matches, removes duplicates
          by setting all but best match to their second best match. 
        
          Inputs
          -----
          best_matches: numpy array, 1D, contains indices of onsets in one set which best 
          match the onsets in another set. 
        
          delta_t_grid: numpy array, 2D, contains all the absolute value time differences
          between all possible pairs of onsets in two sets being compared.
        
          Outputs
          -----
          best_matches: numpy array, 1D. Same as input, but with duplicate matches corrected
        
          delta_t_grid: numpy array, 2D. Same as input, but with values at duplicated 
          positions adjusted to allow finding second best match. 
        
        
          '''
         for i, match_index in enumerate(best_matches):
            if np.isnan(match_index):
              continue
            #check if match index is duplicated
            if len(np.argwhere(best_matches == match_index)) > 1: 
              #create list of indices of duplicates
              duplicates = np.nonzero(best_matches == match_index)[0]
              #find which duplicate has the smallest delta t
              delta_ts = []
              for n in duplicates:
                delta_ts.append(delta_t_grid[int(best_matches[n]), n])
        
              #get all but the best matches of the duplicates
              bad_matches = np.delete(duplicates, np.argmin(delta_ts))
        
              #find second best matches for all but the best duplicate matches
        
              for bad_match in bad_matches:
                delta_t_grid[int(best_matches[bad_match]), bad_match] = np.inf
                best_matches[bad_match] = np.argmin(delta_t_grid[:, bad_match])
                if np.isinf(np.min(delta_t_grid[:, bad_match])):
                  best_matches[bad_match] = np.NaN
         return (best_matches, delta_t_grid)
        
    
    def calc_F1_components_per_file(seg_current_file, 
                                    truth_current_file, 
                                    seg_matches, 
                                    truth_matches, max_gap):
        
        truth_current_file_OG = truth_current_file.copy()
        
        #create dataframe with matches and timestamps for ease of filtering
        matches_and_times = pd.DataFrame({'onsets' : seg_current_file['onsets'], 
                                          'matches' : seg_matches})
        
        #check whether file contains any syllables. If not, skip it.
        if (truth_current_file['labels'] == 's').sum() == 0:
            true_positives = 0
            false_negatives = 0
            false_positives = 0
            return(true_positives, false_positives, false_negatives)
            
        #remove matches before first true syllable, as these likely reflect noise
        first_syllable_time = truth_current_file['onsets'].iloc[np.min(np.nonzero([truth_current_file['labels'] == 's'])[1])]
        matches_and_times = matches_and_times[matches_and_times['onsets'] >= first_syllable_time]
        
        #remove matches after last true syllable, as these likely reflect noise
        last_syllable_time = truth_current_file['offsets'].iloc[np.max(np.nonzero([truth_current_file['labels'] == 's'])[1])]
        matches_and_times = matches_and_times[matches_and_times['onsets'] < last_syllable_time]
        
        #find noise in the middle of the file
        truth_current_file = truth_current_file[(truth_current_file['onsets'] >= first_syllable_time) & 
                                                (truth_current_file['onsets'] < last_syllable_time)]
        middle_noises = np.nonzero([truth_current_file['labels'] == 'n'])[1]
        
        #remove matches that are between the last syllable before noise and the first syllable after noise
        for noise in middle_noises:
            all_sylls_before = truth_current_file[:noise]
            #special case where there is only one syllable before the first noise
            if len(all_sylls_before) == 1:
                last_time_before = all_sylls_before['offsets'].values[0]
            
            #all other cases
            else:
                last_syll_before = np.max(np.nonzero([all_sylls_before['labels'] == 's'])[1])
                last_time_before = all_sylls_before['offsets'].iloc[last_syll_before]
                
            all_sylls_after = truth_current_file[noise:]
            #special case where there is only one syllable after the last noise
            if len(all_sylls_after) == 1:
                first_time_after = all_sylls_after['onsets'].values[0]
            #all other cases
            else:
                first_syll_after = np.min(np.nonzero([all_sylls_after['labels'] == 's'])[1])
                first_time_after = all_sylls_after['onsets'].iloc[first_syll_after]
                
            matches_and_times = matches_and_times[(matches_and_times['onsets'] < last_time_before + max_gap) |
                    (matches_and_times['onsets'] > first_time_after - max_gap)]
            
        true_positives = np.isfinite(matches_and_times['matches']).sum()
        false_positives = np.isnan(matches_and_times['matches']).sum()
        
        #find all true syllables
        syllable_indices = np.nonzero([truth_current_file_OG['labels'] == 's'])[1]
        #select only matches to true syllables (not noise)
        true_matches = truth_matches[syllable_indices]
        #calculate total number of false negatives (true onsets without a match)
        false_negatives = np.isnan(true_matches).sum()
        
        return(true_positives, false_positives, false_negatives)
        
        
class Plot():
    def __init__(self):
        pass
    
    def plot_segmentations(seg_data, seg_label, plot_ground_truth = False, 
                           true_label = 'Ground Truth', 
                           file_idx = 0, figsize = (20, 5), 
                           seg_attribute = 'onsets', plot_title = ""):
        '''
          Generates a spectrogram of the given wave with segmenter onset times 
          and true onset times plotted over top
        
          Inputs
          -----
          truth_current_file: Pandas DataFrame, Subset of a full true syllable table 
          (imported from evsonganaly and cleaned) containing only rows which correspond 
          to the specified wave. 
        
          mfcc_current_file: Pandas DataFrame, Subset of full syllable table which 
          contains only rows which correspond to the specified wave.
        
          wave: numpy array, 1D, Containing sample values from an audio file.
        
          sample_rate: int > 0, the sample rate of the wave in samples per second.
        
          Outputs
          -----
          None
        
          '''
        #retrieve subset of true and seg_tables which correspond to individual file. 
        file_name = np.unique(seg_data.seg_table['files'])[file_idx]
        
        seg_table_current_file = seg_data.seg_table[seg_data.seg_table['files'] == file_name]
        
        #load individual song wave file
        song = dataloading.SongFile(seg_data.song_folder_path + "/" + file_name)
        
        #make spectrogram
        spectrogram = plotting.make_spectrogram(song)
        
        #plot spectrogram -- should this also be a function?
        plotting.plot_spectrogram(spectrogram, song.sample_rate)
        
        if plot_ground_truth:
            
            true_table_current_file = seg_data.true_seg_table[seg_data.true_seg_table['files'] == file_name]
            #select only true syllables, to prevent true noise onsets from being plotted
            true_table_current_file = true_table_current_file[true_table_current_file['labels'] == 's']
            
            #plot seg onsets 
            plt.eventplot(seg_table_current_file[seg_attribute],
                      lineoffsets = 15000, linelengths = 10000, 
                      color = 'white', label = seg_label)
        
            #plot true onsets
            plt.eventplot(true_table_current_file[seg_attribute], 
                      lineoffsets = 5000, linelength = 10000, 
                      color = 'red', label = true_label)
        else:
            #plot seg onsets 
            plt.eventplot(seg_table_current_file[seg_attribute],
                      lineoffsets = 10000, linelengths = 20000, 
                      color = 'white', label = seg_label)
        
        plt.legend()
        plt.title(plot_title)
        plt.show()
        
    
    def plot_seg_criteria(seg_data, segmenter, label, file_idx = 0, figsize = (20, 5),
                          feature_range = (100, 20000)):
        #load a single file
        file_name = np.unique(seg_data.seg_table['files'])[file_idx]
        song = dataloading.SongFile(seg_data.song_folder_path + file_name)
        
        #make spectrogram
        spectrogram = plotting.make_spectrogram(song)
        
        #calculate segmentation criteria
        seg_criteria = segmenter.get_seg_criteria(song)
        #rescale segmentation criteria, so that it is visible over spectrogram
        seg_criteria_scaled = sklearn.preprocessing.minmax_scale(seg_criteria, feature_range = feature_range)
        
        #plot spectrogram
        plotting.plot_spectrogram(spectrogram, song.sample_rate)
        
        x_axis = librosa.frames_to_time(np.arange(len(seg_criteria)), sr = song.sample_rate, 
                                        hop_length = 512, n_fft = 2048)
        plt.plot(x_axis, seg_criteria_scaled, color = 'white', label = label)
        plt.legend()
        
        
        

class Utils:

    def __init__():
        pass
    
    def threshold_optimization(segmenter, Bird_ID, song_folder_path, truth_table_path, 
                               threshold_range, threshold_step, lower_threshold):
        
        thresholds = np.arange(threshold_range[0], threshold_range[1], threshold_step)
        segmentation_scores = pd.DataFrame()
        
        for threshold in thresholds:
            seg_data = segmenter.make_segmentation_table(Bird_ID, song_folder_path, 
                                              upper_threshold = threshold, lower_threshold = lower_threshold)
            seg_data = dataloading.Utils.add_ev_song_truth_table(seg_data, truth_table_path)
            
            seg_data = Metrics.calc_F1(seg_data)
            
            segmentation_score = pd.DataFrame({"F1": [seg_data.F1], 
                                               "precision" : [seg_data.precision],  
                                               "recall" : [seg_data.recall], 
                                               "upper_threshold" : [threshold], 
                                               "lower_threshold" : [lower_threshold]})
            segmentation_scores = segmentation_scores.append(segmentation_score)
            
        optimal_threshold = segmentation_scores['upper_threshold'].iloc[segmentation_scores['F1'].argmax()]
        peak_F1 = segmentation_scores['F1'].max()
        
        return optimal_threshold, peak_F1, segmentation_scores
    
    def calc_F1_many_birds(segmenter, Bird_IDs, 
                           folder_path, upper_threshold, lower_threshold, 
                           truth_table_suffix = "_syll_table.csv"):
        
        segmentation_scores = pd.DataFrame()
        
        segmentations_df = pd.DataFrame()
        
        for Bird_ID in Bird_IDs:
            
            song_folder = folder_path + Bird_ID + "/"
                
            seg_data = segmenter.make_segmentation_table(Bird_ID, song_folder, 
                                              upper_threshold = upper_threshold, 
                                              lower_threshold = lower_threshold)
            seg_data = dataloading.Utils.add_ev_song_truth_table(seg_data, song_folder + Bird_ID + truth_table_suffix)
            
            seg_data = Metrics.calc_F1(seg_data)
            
            segmentation_score = pd.DataFrame({"F1": [seg_data.F1], 
                                               "precision" : [seg_data.precision],  
                                               "recall" : [seg_data.recall], 
                                               "upper_threshold" : [upper_threshold], 
                                               "lower_threshold" : [lower_threshold], 
                                               "Bird_ID" : [Bird_ID]})
            segmentation_scores = segmentation_scores.append(segmentation_score)
            
            segmentation_df = seg_data.seg_table
            segmentation_df["Bird_ID"] = Bird_ID
            segmentations_df = segmentations_df.append(segmentation_df)
            
        return segmentation_scores, segmentations_df
            
        
    def threshold_optimization_many_birds(segmenter, Bird_IDs, folder_path,
                                          threshold_range, threshold_step, lower_threshold, 
                                          truth_table_suffix = "_syll_table.csv"):
        
        segmentation_scores = pd.DataFrame()
        
        for Bird_ID in Bird_IDs:
            song_folder = folder_path + Bird_ID + "/"
            
            truth_table_path = song_folder + Bird_ID + truth_table_suffix
            
            optimal_thresh, peak_F1, threshold_table = Utils.threshold_optimization(segmenter, Bird_ID, 
                                                                                    song_folder, truth_table_path,
                                                                                    threshold_range, 
                                                                                    threshold_step, 
                                                                                    lower_threshold)
            threshold_table['Bird_ID'] = Bird_ID
            segmentation_scores = segmentation_scores.append(threshold_table)
            
        mean_F1s = segmentation_scores.groupby("upper_threshold").mean()
        peak_mean_F1 = mean_F1s.F1.max()
        
        mean_F1s.reset_index(inplace = True)
        optimal_threshold = mean_F1s.upper_threshold.iloc[mean_F1s.F1.argmax()]
        
        return optimal_threshold, peak_mean_F1, segmentation_scores
    
    def plot_segmentations_many_birds(segmenter, Bird_IDs, folder_path, seg_label,
                                      upper_threshold, lower_threshold,
                                      plot_ground_truth = False,
                                      files_per_bird = 3, random_seed = 2021, 
                                      true_label = "Ground Truth", figsize = (20, 5), 
                                      seg_attribute = 'onsets', truth_table_suffix = "_syll_table.csv"):
        
        for Bird_ID in Bird_IDs:
            song_folder = folder_path + Bird_ID + "/"
            
            truth_table_path = song_folder + Bird_ID + truth_table_suffix
            
            seg_data = segmenter.make_segmentation_table(Bird_ID, song_folder, 
                                                         upper_threshold, 
                                                         lower_threshold)
            
            if plot_ground_truth:
                seg_data = dataloading.Utils.add_ev_song_truth_table(seg_data, truth_table_path)
            
            all_files = np.unique(seg_data.seg_table.files)
            
            random.seed(random_seed)
            indices_to_plot = random.sample(range(0, len(all_files)), files_per_bird)
            
            for rand_index in indices_to_plot:
                
                Plot.plot_segmentations(seg_data, seg_label = seg_label, 
                                        plot_ground_truth = plot_ground_truth,
                                    true_label = true_label, figsize = figsize, 
                                    seg_attribute = seg_attribute, file_idx = rand_index, 
                                    plot_title = all_files[rand_index])
                