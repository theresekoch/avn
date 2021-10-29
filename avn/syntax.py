# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 10:49:15 2021

@author: Therese
"""
import avn.dataloading as dataloading
import numpy as np
import pandas as pd
import scipy.signal
from statistics import median
from statistics import mean
from itertools import groupby
import more_itertools as mit
import math
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import avn
import warnings

class SyntaxData: 
    
    def __init__(self, Bird_ID, syll_df):
        """
        Parameters
        ----------
        Bird_ID : str
            String containing a unique identifier for the subject bird. 
        syll_df : pandas DataFrame
            pandas dataframe containing one row for every syllable to be analyzed
            from the subject bird. It must contain columns *onsets* and *offsets* 
            which contain the timestamp in seconds at which the syllable occurs 
            within a file, *files* which contains the name of the .wav file in 
            which the syllable is found, and *labels* which contains a categorical
            label for the syllable type. These can be generated through manual song
            annotation, or automated labeling methods. 

        """
        
        self.Bird_ID = Bird_ID
        self.syll_df = syll_df
        
        #make sure there are no negative onset times in syll_df
        #by replacing all negative onsets with an onset time of 0.
        self.syll_df['onsets'] = self.syll_df.onsets.where(syll_df.onsets > 0, 0)

        #get unique syllable labels before file boundaries are added
        self.unique_labels = self.syll_df.labels.unique().tolist()
        
        self.file_bounds_added = False
        self.gaps_added = False
        self.min_gap = None
        self.calls_dropped = False
        
    def add_file_bounds(self, song_folder_path):
        """
        Add rows representing syllable boundaries to self.syll_df. A new row 
        with label value 'file_start' and onset and offset values = 0 will be 
        added before the first syllable of a new file and a new row with label 
        value 'file_end' and onset and offset values reflecting the duration of 
        the file in question will be added after the last syllable of a file. 

        Parameters
        ----------
        song_folder_path : str
            Path to folder containing .wav files of songs in SyntaxData.syll_df. 
            Should end with '/'.

        Raises
        ------
        RuntimeError
            If file bounds have already been added to this SyntaxData object, 
            this error is raised to inform the user that file bounds will not be
            added a second time. This is based on the value of the boolean 
            self.file_bounds_added. 

        Returns
        -------
        None.

        """
        
        #check whether file bounds have already been added. If so, no need 
        #to do it again. 
        if  self.file_bounds_added == True: 
            raise RuntimeError("File bounds have already been added. This process will not be repeated.")
        
        #get unique syllable labels before file boundaries are added
        self.unique_labels = self.syll_df.labels.unique().tolist()
        
        #add syllable boundaries to syll_df
        for file_name in np.unique(self.syll_df.files):
            #load song file
            file_path = song_folder_path + self.Bird_ID + '/' + file_name
            song = dataloading.SongFile(file_path)
            
            #create df for current file with 'file_start' and 'file_end' rows
            curr_file_df = pd.DataFrame({'files': file_name, 
                                         'onsets': [0, song.duration], 
                                         'offsets': [0, song.duration], 
                                         'labels' : ['file_start', 'file_end']})
            
            #append dataframe with file start and file end to syllable dataframe
            self.syll_df = pd.concat([self.syll_df, curr_file_df])
            
        #specify the order of labels so that 'file_stat' comes before other syllables
        #with onset timestamp 0. 
        self.syll_df['labels'] = self.syll_df['labels'].astype('category')
        category_order = ['file_start'] + self.unique_labels + ['file_end']
        self.syll_df['labels'] = self.syll_df['labels'].cat.set_categories(category_order, ordered = True)
        
        #sort row order so file boundaries are placed correctly
        self.syll_df = self.syll_df.sort_values(['files', 'onsets', 'labels'])
        
        #record that file boundaries have been added. 
        self.file_bounds_added = True
            
    def get_gaps_df(self):
        """
        Makes a dataframe with all the gaps between syllables in self.syll_df

        Returns
        -------
        gaps_df : Pandas DataFrame
            Dataframe with columns *files*, *onsets*, *offsets*, *labels*, and *duration*
            which represents each gap as a single row. Onsets and offsets give
            the timestamps in seconds at which the gap occurs, and duration 
            gives the duration of the gap in seconds. The label for all gaps 
            is 'silent_gap'.

        """
        #This will only work correctly if there are no gaps in syll_df already. 
        #so let's filter them out
        syll_df_no_gaps = self.syll_df[self.syll_df.labels != 'silent_gap']

        #get gap offsets - which are just syll_df onsets shifted down by one
        offsets = syll_df_no_gaps.onsets[1:].tolist() + [np.NaN]
        
        #create dataframe with all gaps
        gaps_df = pd.DataFrame({'files' : syll_df_no_gaps.files, 
                                'onsets' : syll_df_no_gaps.offsets,
                                'offsets': offsets, 
                                'labels': 'silent_gap'})
        #calculate gap durations
        gaps_df['durations'] = gaps_df.offsets - gaps_df.onsets
        
        #drop any gaps with durations < 0. These reflect file boundaries or 
        #segmentation errors.
        gaps_df = gaps_df[gaps_df.durations >=0 ]
        
        return gaps_df
        
        
    def add_gaps(self, min_gap = 0.2):
        """
        Add rows representing silent gaps between syllables longer than `min_gap`
        to self.syll_df. 

        Parameters
        ----------
        min_gap : float, optional
            Minimum duration in seconds for a gap between syllables to be 
            considered syntactically relevant. This value should be selected 
            such that gaps between syllables in a bout are shorter than min_gap, 
            but gaps between bouts are longer than min_gap. The default is 0.2.

        Raises
        ------
        RuntimeError
            If file bounds have already been added to this SyntaxData object, 
            this error is raised to inform the user that file bounds will not be
            added a second time. This is based on the value of the boolean 
            self.file_bounds_added. 

        Returns
        -------
        None.

        """
        #check whether gaps have already been added
        if  self.gaps_added == True: 
            raise RuntimeError("Gaps have already been added. This process will not be repeated.")
        #check whether file bounds have been added
        if self.file_bounds_added == False:
            raise RuntimeError("This function requires that file bounds be added to `.syll_df`. To do this, call `.add_file_bounds()`.")
        
        #make dataframe with all gaps and their durations
        gaps_df = self.get_gaps_df()
        
        #select only gaps that are longer than min_gap
        long_gaps_df = gaps_df[gaps_df.durations > min_gap]
        
        #add long gaps to syll_df
        self.syll_df = pd.concat([self.syll_df, long_gaps_df])
        
        #specify order for labels so that rows with the same onset times (ie a 
        #gap at the start of a file) are ordered correctly
        category_order  =['file_start'] + self.unique_labels + ['silent_gap', 'file_end']
        self.syll_df['labels'] = self.syll_df['labels'].astype('category')
        self.syll_df['labels'] = self.syll_df['labels'].cat.set_categories(category_order, ordered = True)
        
        #sort rows so that gaps are in the correct positions
        self.syll_df = self.syll_df.sort_values(['files', 'onsets', 'labels'])
        
        #record that gaps have been added. 
        self.gaps_added = True
        self.min_gap = min_gap
        
    def drop_calls(self):
        """
        This function drops any rows in self.syll_df reflecting syllables that 
        are preceeded and followed by silent gaps, as these likely reflect calls. 

        Raises
        ------
        RuntimeError
            Gaps must be added to self.syll_df before calls can be identified, 
            so this function will raise an error if gaps have not been added. 
            It will also raise an error if calls have already been dropped from 
            self.syll_df, to avoid repeating this process unnecessarily. 

        Returns
        -------
        None.

        """
        
        #check whether gaps have been added
        if self.gaps_added == False:
            raise RuntimeError('Gaps must be added before calls can be identified. Please run .add_gaps() before proceeding.')
        #check whether calls have already been dropped
        if self.calls_dropped == True:
            raise RuntimeError("Calls have already been dropped. This process will not be repeated.")
        
        #save copy of syll_df with calls included for later use (with .get_prop_sylls_in_short_bouts(), for example)
        self.syll_df_with_calls = self.syll_df

        #create dataframe with the preceeding and following syllable for every 
        #syllable in syll_df
        syll_trios = pd.DataFrame({'prev': self.syll_df.labels.shift(1), 
                                   'curr' : self.syll_df.labels,
                                   'next' : self.syll_df.labels.shift(-1)})
        
        #filter out all rows which have 'silent_gap' as the preceeding and following syllable
        self.syll_df = self.syll_df[~((syll_trios.prev == 'silent_gap') & (syll_trios.next == 'silent_gap'))]
        
        #record that calls have been dropped
        self.calls_dropped = True
        
    def make_transition_matrix(self):
        """
        This funtion calculates the first order transition matrix of syllables
        in syll_df. It creates 2 new attributes to self; self.trans_mat which
        contains the raw counts of each transition between syllables types and 
        self.trans_mat_prob which contains the conditional probability of a 
        transition, given that a particular syllable was just produced. 

        Returns
        -------
        None.

        """
        #Add file bounds if they have not already been added
        if self.file_bounds_added == False: 
            raise RuntimeError('This function requires that file boundaries already be added to `.syll_df`. To do this, first call `.add_file_bounds()`.')
        
        if self.gaps_added == False: 
            raise RuntimeError('This function requires that gaps already be added to `.syll_df`. To do this, first call `.add_gaps()`.')
        
        #make dataframe containing pairwise transitions between syllables
        transitions_df = pd.DataFrame({'syll_1' : self.syll_df.labels, 
                                       'syll_2' : self.syll_df.labels.shift(-1), 
                                       'count' : 1})
        
        #make transition matrix with raw transition counts
        trans_mat = transitions_df.groupby(['syll_1', 'syll_2']).count().unstack()
        
        #drop file_start and file_end rows and columns as these are not biologically
        #relevant states
        trans_mat.columns = trans_mat.columns.droplevel(0)
        trans_mat = trans_mat.drop(index = ['file_end', 'file_start'], 
                                   columns = ['file_end', 'file_start'])
        
        #set the cell for transitions from silence to silence to 0. Any values 
        #here are artifacts of removing calls. 
        trans_mat.loc['silent_gap', ['silent_gap']] = 0
        
        #divide each row by its sum to get the conditional probabilities
        trans_mat_prob = trans_mat.div(trans_mat.sum(axis = 1), axis = 0)
        
        self.trans_mat = trans_mat
        self.trans_mat_prob = trans_mat_prob
        
    def get_entropy_rate(self):
        """
        Calculates the entropy rate of bird's syntax based on transition matrices. 
        For more information on entropy rate, refer to online documentation. 

        Returns
        -------
        entropy_rate : float
            Entropy rate of syntax summarised in self.trans_mat. bounded by 0 and 
            log_2(number of unique syllable states), where larger values reflect 
            more variable / unpredictable syntax. 

        """
        #calculate the shannon entropy of each row ie each syllable of the 
        #transition matrix probabilities
        syll_entropies = scipy.stats.entropy(self.trans_mat_prob, axis = 1, base = 2)
        
        #calculate the static distribution of each syllable based on how frequently
        #it occurs in the raw count transition matrix
        emperical_static = self.trans_mat.sum(axis = 1) / self.trans_mat.sum().sum()
        
        #calculate the entropy rate
        entropy_rate = (syll_entropies * emperical_static).sum()
        
        return entropy_rate
        
    def get_prob_repetitions(self):
        """
        Find the probability of transition to self (ie repetition) for each 
        syllable type, and return values in a dataframe. 

        Returns
        -------
        prob_repetition_df : pandas DataFrame
            DataFrame with columns *Bird_ID*, *syllable* and *prob_repetition*
            containing the probability that each syllable type produced by this 
            bird transitions to itself base on the song data in self.syll_df. 

        """
        #make empty dataframe to store data
        prob_repetition_df = pd.DataFrame()
        
        #loop over each syllable type:
        for syllable in self.trans_mat_prob.index.values:
            
            #find the probability of a syllable transitioning to itself
            prob_repetition = self.trans_mat_prob.loc[syllable, syllable]
            
            #add transition probability to dataframe
            curr_df = pd.DataFrame({'Bird_ID': [self.Bird_ID], 
                                    'syllable' : [syllable], 
                                    'prob_repetition' : [prob_repetition]})
            prob_repetition_df = pd.concat([prob_repetition_df, curr_df])
                
        return prob_repetition_df    
    
    
    def get_single_repetition_stats(self):
        """
        Analyzes repetitions of single syllables. Specifically looks at
        occurances of repetition bouts of different durations 
        (ie 2 identical syllables in a row, 3 identical syllables in a row, etc.). 

        Returns
        -------
        rep_count_df: Pandas DataFrame
            Dataframe containing counts of syllable occurances in repetition bouts 
            of different durations for every syllable type in self.syll_df.labels. 
            There is one row per syllable and column names refer to duration of 
            repetition bout. 1 = syllable produced but not repeated. 2 = syllable 
            repeated twice in a row only etc. 

        rep_stats_df: Pandas DataFrame
            DataFrame containing statistics about repetition bout length.
            Columns contain the mean_bout_length, median_bout_length, and CV_bout_length,
            where bout refers to a repetition bout, ie an instance of the 
            same syllable being repeated many times. These values can be 
            used for identifying introductory notes and/or abnormally repeated song syllables. 

        """
        #raise warning if file bounds or gaps haven't been added
        if self.file_bounds_added == False:
            warnings.warn("It is recommended to add file bounds to `.syll_df` before getting single repetition stats. To do this, first run `.add_file_bounds()`.")
        if self.gaps_added == False:
            warnings.warn("It is recommended to add gaps to `.syll_df` before getting single repetition stats. To do this, first run `.add_gaps()`.")


        #create dictionary with list of repetition bout durations for each syllable
        rep_dict = self._make_single_rep_dictionary()
        
        #convert dictionary of repetition bout durations to df with counts of 
        #occurances of repetition bouts of different durations
        rep_count_df = self._rep_dict_to_count_df(rep_dict)
        
        #drop meaningless syllable types from count dataframe
        rep_count_df = self._drop_meaningless_single_reps(rep_count_df)
        
        #calculate summary statistics fon repetition bout duration for 
        #each syllable and return as df
        rep_stats_df = self._calc_repetition_stats(rep_dict)
        
        #drop meaningless syllable types from stats dataframe
        rep_stats_df = self._drop_meaningless_single_reps(rep_stats_df)
        
        return rep_count_df, rep_stats_df
        
    def get_pair_repetition_stats(self):
        """
        Analogous to `self.get_single_repetition_stats`, but with repetitions
        of a syllable pair, rather than a syllable type. For example, 
        the sequence 'ababab' reflects a repetition bout of duration 3 
        for the syllable pair 'a' and 'b'. 

        Returns
        -------
        rep_count_df: Pandas DataFrame
            Dataframe containing counts of syllable pair occurances in repetition bouts 
            of different durations for every syllable pair occuring in self.syll_df.labels. 
            There is one row per syllable pair and column names refer to duration of 
            repetition bout. 1 = syllable pair produced but not repeated (eg iabcd). 2 = syllable 
            pair repeated twice in a row only (eg iababcd) etc. 

        rep_stats_df: Pandas DataFrame
            DataFrame containing statistics about repetition bout length.
            Columns contain the mean_bout_length, median_bout_length, and CV_bout_length,
            where bout refers to a repetition bout, ie an instance of the 
            same syllable pair being repeated many times. These values can be 
            used for identifying  abnormally repeated song syllables. 
        
        """
        #raise warning if file bounds or gaps haven't been added
        if self.file_bounds_added == False:
            warnings.warn("It is recommended to add file bounds to `.syll_df` before getting pair repetition stats. To do this, first run `.add_file_bounds()`.")
        if self.gaps_added == False:
            warnings.warn("It is recommended to add gaps to `.syll_df` before getting pair repetition stats. To do this, first run `.add_gaps()`.")

        #create dictionary with list of repetition bout durations for each syllable pair
        rep_dict = self._make_pair_rep_dictionary()
        
        #convert dictionary of repetition bout durations to df with counts of 
        #occurances of repetition bouts of different durations
        rep_count_df = self._rep_dict_to_count_df(rep_dict)
        #add columns with first syllable and second syllable in pair separated
        rep_count_df = self._split_pair_label(rep_count_df)
        
        #drop meaningless syllable types from count dataframe
        rep_count_df = self._drop_meaningless_pair_reps(rep_count_df)
        
        #calculate summary statistics fon repetition bout duration for 
        #each syllable pair and return as df
        rep_stats_df = self._calc_repetition_stats(rep_dict)
        #add columns with first syllable and second syllable in pair separated
        rep_stats_df = self._split_pair_label(rep_stats_df)
        
        #drop meaningless syllable types from stats dataframe
        rep_stats_df = self._drop_meaningless_pair_reps(rep_stats_df)
        
        return rep_count_df, rep_stats_df
        
    def _make_rep_dictionary(self, label_list):
        """
        Create a dictionary with unique elements of `label_list` as keys 
        and a list of repetition bout durations for every occurance of 
        the element in `label_list` as values. 
        
        Note that `label_list` could contain individual syllable labels
        or pairs of syllable labels. 

        Parameters
        ----------
        label_list: list
            List of syllable labels or pairs of syllable labels in the 
            order they were produced. 

        Returns
        -------
        d: dictionary
            Dictionary with unique elements of `label_list` as keys 
            and a list of repetition bout durations for every occurance
            of the element in `label_list` as values. 
        
        """
        #create dictionary with list of repetition durations for label in label_list
        d = dict()
        for key, values in groupby(label_list):
            d.setdefault(key, []).append(len(list(values)))
            
        return d
    
    
    def _make_single_rep_dictionary(self):
        """
        Create dictionary with syllable types as keys and a list of repetition
        bout durations for every occurance of the of the syllable as values. 

        The output can be converted to a dataframe of counts of repetition bouts of
        different durations with `self._rep_dict_to_count_df()`

        Returns
        -------
        d: dictionary 
            Dictionary with syllable types in self.syll_df.labels as keys and 
            a list of repetition bout durations for every occurance of the 
            syllable as values. 

        """
        #convert syll_df labels to list of syllable labels 
        label_list = self.syll_df.labels.values.to_list()
        #make dictionary of repetition bout durations.
        d = self._make_rep_dictionary(label_list)
        
        return d
    
    def _combine_rep_count_dictionaries(self, dict_1, dict_2):
        """
        Combine dictionaries of syllable pair repetitions calculated 
        on odd indices and even indices. See self._make_pair_rep_dictionary()
        for context. 

        Parameters
        ----------

        dict_1: dictionary
            Dictionary with syllable pairs as keys and a list of repetition bout 
            durations as values, where pairs are determine starting at an 
            index of either 0 or 1. 

        dict_2: dictionary
            Dictionary with syllable pairs as keys and a list of repetition bout
            durations as values, where pairs are determine starting at an index 
            different from that in dict_1. 

        Returns
        -------

        combo_dict: dictionary
            Dictionary with syllable pairs as keys and a list of repetition 
            bout durations for every occurance of the syllable pair, combined
            across pairs starting at even indices or an odd indices in self.syll_df.labels. 
        
        """

        #initialize empty dictionary
        combo_dict = dict()

        #loop through each syllable pair key in dict_1
        for key in dict_1.keys():
            #if the key is in dict 1 and dict 2, then concatenate the values 
            #and add entry to combo dict
            if key in dict_2.keys():
                combo_dict[key] = dict_1[key] + dict_2[key]

            #if the key is in dict 1 but not in dict 2, copy the entry
            #from dict 1 to the combo dict
            else:
                combo_dict[key] = dict_1[key]

        #loop through each syllable pair key in dict_2
        for key in dict_2.keys():
            #if the key is in dict 2 but not in dict 1, copy the 
            #entry from dict 2 to combo dict. 
            #if the key is in both, it is accounted for when looping 
            #through dict 1. 
            if not (key in dict_1.keys()):
                combo_dict[key] = dict_2[key]
                
        return combo_dict
        
    def _make_pair_rep_dictionary(self):
        """
        Create dictionary with syllable pairs as keys and a list of repetition
        bout durations for every occurance of the of the syllable pair as values. 

        The output can be converted to a dataframe of counts of repetition bouts of
        different durations with `self._rep_dict_to_count_df()`

        Returns
        -------
        combo_dict: dictionary
            Dictionary with syllable pairs as keys and a list of repetition bout 
            durations for every occurance of the syllable pair as values. 
        
        """
        #make lists with only every other syllable label
        even_labels = self.syll_df.labels[::2]
        odd_labels = self.syll_df.labels[1::2]
        
        #get label pairs starting at index 0 and index 1
        label_pairs = [str(x) + "_" + str(y) for x, y in zip(even_labels, odd_labels)]
        label_pairs_shifted = [str(x) + "_" + str(y) for x, y in zip(odd_labels, even_labels.shift(-1))]
        
        #create dictionary with label pairs starting at index 0
        even_dict = self._make_rep_dictionary(label_pairs)
        #create dictionary with label pairs starting at index 1
        odd_dict = self._make_rep_dictionary(label_pairs_shifted)
        
        #combine index 0 and index 1 pair counts into a single dictionary
        combo_dict = self._combine_rep_count_dictionaries(even_dict, odd_dict)
        
        return combo_dict
    
    def _rep_dict_to_count_df(self, rep_dict):
        """
        Converts a dictionary as returned by `_make_pair_rep_dictionary()` 
        or `_make_single_rep_dictionary` into a dataframe with 
        the count of occurances of repetitions bout of different lengths
        for each unique syllable or syllable pair in `rep_dict`. 

        Parameters
        ----------

        rep_dict: dictionary
            Dictionary with unique syllable or syllable pairs as keys 
            and lists containing repetition bout durations for every 
            occurance of the syllable or syllable pair as values. 
            Dictionaries of this type are returned by `_make_pair_rep_dictionary()`
            for syllable pairs and `_make_single_rep_dictionary()` for 
            individual syllables. 

        Returns
        -------

        rep_count_df: Pandas DataFrame
            Dataframe containing counts of syllable or syllable pair occurances 
            in repetition bouts of different durations as stored in `rep_dict`. 
            There is one row per syllable or syllable pair and column names refer 
            to the duration of repetition bout. 
            1 = syllable or pair produced but not repeated. 2 = syllable or pair
            repeated twice in a row only etc. 

        """
        #initialize empty dataframe
        rep_count_df = pd.DataFrame()
        
        #count occurances of repetition bouts with different durations and 
        #record them in dataframe
        for syllable in rep_dict.keys():
            value_counts = pd.DataFrame(pd.Series(rep_dict[syllable]).value_counts()).transpose()
            value_counts['syllable'] = [syllable]
            
            rep_count_df = pd.concat([rep_count_df, value_counts])
            
        #add Bird_ID column to dataframe
        rep_count_df['Bird_ID'] = self.Bird_ID
        
        #fill nan values with zeros
        rep_count_df = rep_count_df.fillna(0)
        
        return rep_count_df
        
    def _split_pair_label(self, rep_df):
        """
        Adds columns 'first_syll' and 'second_syll' to `rep_df`, containing 
        the labels of the first and second syllable types in a syllable pair. 
        This facilitates subsequent filtering of syllable pairs. 

        Parameters
        ----------
        rep_df: Pandas DataFrame
            Dataframe containing counts of syllable pair occurances 
            in repetition bouts of different durations. 
            There is one row per syllable pair and column names refer 
            to the duration of repetition bout. 
            1 = syllable pair produced but not repeated. 2 = syllable pair
            repeated twice in a row only etc. 'syllable' column contains pair 
            of syllables separated by '_' (for example 'a_b' for the pair of 
            syllables 'a' and 'b'). 

        Returns
        -------

        rep_df: Pandas DataFrame
            Dataframe containing counts of syllable pair occurances
            in repetition bouts of different durations. Identical to 
            the input `rep_df`, but with columns 'first_syll' and 'second_syll'
            added with labels of the first and second syllables of the syllable 
            pair, respectively. 

        """
        
        rep_df['first_syll'] = rep_df.syllable.str.split('_', expand = True)[0]
        rep_df['second_syll'] = rep_df.syllable.str.split('_', expand = True)[1]
        
        return rep_df
    
    def _calc_repetition_stats(self, rep_dict):

        """
        Creates a dataframe of repetition bout duration distribution 
        summary statistics for each key in `rep_dict`. 

        Parameters
        ----------
        rep_dict: dictionary
            Dictionary with unique syllable or syllable pairs as keys 
            and lists containing repetition bout durations for every 
            occurance of the syllable or syllable pair as values. 
            Dictionaries of this type are returned by `_make_pair_rep_dictionary()`
            for syllable pairs and `_make_single_rep_dictionary()` for 
            individual syllables. 

        Returns
        -------
        rep_stats_df: Pandas DataFrame
            DataFrame containing statistics about repetition bout length.
            Columns contain the mean_bout_length, median_bout_length, and CV_bout_length,
            where bout refers to a repetition bout, ie an instance of the 
            same syllable or syllable pair being repeated many times in a row. 
            These values can be used for identifying abnormally repeated song syllables. 
        """
        
        #initialize empty dataframe in which to record repetition stats
        rep_stats_df = pd.DataFrame()
        
        #loop over each syllable or syllable pair type in rep_dict
        for syllable in rep_dict.keys():
            
            #calculate repetition stats for given syllable
            mean_reps = mean(rep_dict[syllable])
            median_reps = median(rep_dict[syllable])
            CV_reps = scipy.stats.variation(rep_dict[syllable])
            
            #package repetition stats into dataframe
            curr_df = pd.DataFrame({'syllable' : [syllable], 
                                    'mean_repetition_length' : mean_reps, 
                                    'median_repetition_length' : median_reps, 
                                     'CV_repetition_length' : CV_reps})
            #append df for current syllable to df with all syllables
            rep_stats_df = pd.concat([rep_stats_df, curr_df])
            
        #add column with Bird_ID
        rep_stats_df['Bird_ID'] = self.Bird_ID
            
        return rep_stats_df
            
    def _drop_meaningless_pair_reps(self, rep_df):

        """
        Returns a copy of `rep_df` with rows containing silence or file boundaries
        as the first or second syllable removed, as well as rows where the first and 
        second syllable of the pair are identical (this form of repetition is 
        better studied with `get_single_repetition_stats()). 

        Parameters
        ----------
        rep_df: Pandas DataFrame
            Dataframe containing repetition statistics or counts for syllable pairs. 
            Must have the syllable pair label split across 2 columns by 
            `_spilt_pair_label()`. 

        Returns
        -------

        rep_df: Pandas DataFrame
            Copy of input `rep_df` with rows containing silent gaps or file boundaries
            as the first or recond syllable of the pair removed, as well as rows where
            the first and second syllable of the pair are identical removed. 
        
        """
        
        #drop pairs where the first or second syllable is a file boundary or silent gap
        rep_df = rep_df[~rep_df.first_syll.isin(['silent', 'file'])]
        rep_df = rep_df[~rep_df.second_syll.isin(['silent', 'file'])]
        
        #drop pairs where the first and second syllable are the same. 
        #stats concerning a single syllable repeating are best asessed with 
        #`.get_single_repetition_stats()`. 
        rep_df = rep_df[rep_df.first_syll != rep_df.second_syll]
        
        
        return rep_df
    
    def _drop_meaningless_single_reps(self, rep_df):

        """
        Returns a copy of `rep_df` with rows reflecting non-meaningful label types 
        (silent gaps and file boundaries) removed. 

        Parameters
        ----------
        rep_df: Pandas DataFrame
            Dataframe containing repetition statistics or counts for individual 
            syllables. 

        Returns
        -------
        rep_df: Pandas DataFrame
            Copy of input `rep_df` with rows  reflecting repetitions of silent gaps 
            or file boundaries removed. 

        
        """
        
        #filter out rows reflecting file boundaries or a silent gap
        rep_df = rep_df[~rep_df.syllable.isin(['file_start', 
                                               'file_end', 
                                               'silent_gap'])]
        return rep_df
        
        
    def make_syntax_raster(self, alignment_syllable = None, sort_bouts = True):
        """
        Create a dataframe where each row reflects a song bout (a sequence of syllables flanked by 
        file boundaries or long silent gaps), and each cell contains the label of the song syllable
        produced at that index in the song bout. This can be plotted using `plot_syntax_raster()`
        to get an view of song syntax variability from the subject bird. 

        Parameters
        ----------
        alignment_syllable: string, optional
            The alignment syllable should correspond to a syllable label in `self.syll_df.labels`. 
            If provided, song bouts will be aligned such that the first occurance of the 
            alignment syllable happens at the same index across bouts. This can make it 
            easier to detect patterns in syntax across bouts. It is generally best to set 
            the alignment syllable to be the first syllable of the dominant song motif, 
            following any intro notes. 

        sort_bouts: bool, optional
            If True, bouts will be sorted such that bouts with more similar sequences will occupy
            sequential rows in `syntax_raster_df`. This can make it easier to detect syntax patterns
            agnostic to the order in which bouts were produced. If False, the order of bouts in 
            `syntax_raster_df` will be the order in which the bouts occur in `self.syll_df`. 
            The default is True. 

        Returns
        -------
        syntax_raster_df: Pandas DataFrame
            Dataframe where each row reflects a song bout (a sequence of syllables flanked by 
            file boundaries or long silent gaps), and each cell contains the label of the song 
            syllable produced at that index in the song bout, based on `self.syll_df.labels`. The
            number of columns depends on the length of the longest song bout. This can be plotted using 
            `plot_syntax_raster()` to get a view of song syntx variability from the subject bird. 

        """
        #check whether file boundaries and gaps have been added yet, as they are required for setting bout boundaries
        if self.file_bounds_added == False:
            raise  RuntimeError('This function requires that file boundaries already be added to `.syll_df`. To do this, first call `.add_file_bounds()`.')
        if self.gaps_added == False:
            raise RuntimeError('This function requires that gaps already be added to `.syll_df`. To do this, first call `.add_gaps()`.')

        #restructure sequence of syllables to list of bouts
        bouts_list = self._syllables_to_bouts(self.syll_df.labels.to_list())

        #align bouts to alignment syllable if alignment syllable is not None
        if not (alignment_syllable == None):
            bouts_list, max_first_index = self._align_bouts_to_syllable(bouts_list, alignment_syllable)

            #sort bouts so that similar bouts are together if sort_bouts == True
            if sort_bouts == True:
                bouts_list = self._sort_bouts(bouts_list, max_first_index)
        
        #sort bouts without aligning first if no alignment syllable is provided and if sort_bouts == True
        else:
            if sort_bouts == True:
                bouts_list = self._sort_bouts(bouts_list)

        #convert bouts list to dataframe
        syntax_raster_df = pd.DataFrame(bouts_list)

        return syntax_raster_df

    def _syllables_to_bouts(self, syll_labels):
        """
        Converts list of syllables into list of bouts by splitting at file boundaries and 
        long silent gaps. Each bout is itself a list of syllable labels.

        Parameters
        ----------
        syll_labels: list 
            List of syllable labels including file boundaries and long silent gaps. 

        Returns
        -------
        bout_list: list of lists
            List of lists, where each sub-list reflects a song bout (a sequence of syllables
            flanked by file boundaries or long silent gaps), and contains a sequence of 
            syllable labels reflecting the syllables in the bout. 
        
        """

        #split sequence of syllables at every 'file_start', 'file_end' and 'silent_gap' to get 
        #a list of lists where each sublist is a bout
        bout_list = mit.split_at(syll_labels, pred = lambda x: x in ['file_start', 'file_end', 'silent_gap'])

        #drop all bouts containing no syllables. These appear when file boundaries or silent gaps appear back-to-back
        bout_list = [x for x in bout_list if len(x) > 0]

        return bout_list

    def _align_bouts_to_syllable(self, bout_list, alignment_syll):
        """
        Pads bouts in  `bout_list` with nan values such that the first occurance of 
        the `alignment_syll` always happend at the same index, `max_first_index`. 
        Selecting a good alignment syllable (generally the first note of a song motif)
        can make patterns appear more clearly in a syntax raster plot. 

        Parameters
        ----------
        bout_list: list of lists
            List of lists, where each sublist reflects a song bout (a sequence of syllables 
            flanked by file boundaries or long silent gaps), and contains a sequence of 
            syllable labels reflecting the syllables in the bout. This can be created from 
            a list of syllable labels with file boundaries and gaps included using 
            `_syllables_to_bouts()`. 

        alignment_syll: String, int or char
            The label of the syllable to be used to align bouts. This must be one of 
            the labels in self.syll_df.labels. 
            All bouts will be padded with nan values before the first syllable such that 
            the first occurance of the alignment syllable will occur at the same index 
            across bouts. It is generally best to pick an alignment syllable that reflects
            the first syllable of the dominant song motif. 

        Returns
        -------
        bout_list_padded: list of lists
            Copy of input `bout_list`, where each bout containing the alignment_syll
            is padded with nan values at the beginning so that the first occurance of 
            the alignment_syll happens at the same index across bouts. 

        max_first_index: int
            Index of the first occurance of the alignment_syll in every bout. 
        
        """
        #check whether alignment_syll is a valid label
        if not alignment_syll in self.unique_labels: 
            raise ValueError(str(alignment_syll) + " is not a valid syllable label. Valid options are :" + str(self.unique_labels))

        #initialize empty list to store first indices of the alignment syllable in each bout
        first_indices = []

        #for every bout, find the first index where the alignment syllable appears, if it is in the bout. 
        for bout in bout_list: 

            if alignment_syll in bout: 
                first_index = bout.index(alignment_syll)
            else: 
                first_index = np.nan
            
            #append first_index for bout to list of first indices for all bouts
            first_indices.append(first_index)
            
        #find the latest first index of the alignment syllable across bouts. 
        #all other bouts will be padded so the alignment syllable occurs for 
        #the first time at this index
        max_first_index = np.nanmax(first_indices)

        #initialize empty list to store new padded bouts
        bout_list_padded = []

        #loop over each bout and the first index at which the alignment syllable occurs in that bout. 
        for bout, first_index in zip(bout_list, first_indices):
            #if the alignment syllable exists in the bout, pad the bout so that the new first_index == max_first_index
            if not math.isnan(first_index):
                #calculate how many nan values must be added to the start of the bout so that first_index shifts
                #to max_first_index
                to_pad = max_first_index - first_index
                #create list of nan values of length `to_pad` to add to the start of the bout
                padding = [np.nan for x in np.arange(int(to_pad))]
                #add padding
                new_bout = padding + bout

                #append new padded bout to list of padded bouts
                bout_list_padded.append(new_bout)

            #if the alignment syllable is not in the bout, don't pad it. 
            else: 
                bout_list_padded.append(bout)

        return bout_list_padded, max_first_index

    def _sort_bouts(self, bout_list, max_first_index = None):
        """
        Reorder bouts such that bouts with similar syllable sequence are together. 
        This makes it impossible to see patterns in bout syntax over time, but does 
        make it easier to detect patterns agnostic to the order in which the bouts are
        produced. 

        Parameters
        ----------
        bout_list: list of lists
            List of lists, where each sublist reflects a song bout (a sequence of syllables 
            flanked by file boundaries or long silent gaps), and contains a sequence of 
            syllable labels reflecting the syllables in the bout. This can be created from 
            a list of syllable labels with file boundaries and gaps included using 
            `_syllables_to_bouts()`.

        max_first_index: int, optional
            If a value for max_first_index is provided, bouts will be sorted according to syllable 
            composition at and following `max_first_index`. Otherwise they will be sorted 
            alphanumerically starting with the first index. If bouts are aligned to an 
            alignment syllable using `_align_bouts_to_syllable`, it is recommended to provide
            the `max_first_idex` value returned by that function. 

        Returns
        -------
        bout_list: list of lists
            List of lists similar to the input `bout_list`, but the order of bouts within the 
            list is sorted alphanumerically based on the sequence of syllables within the bout. 

        """

        #convert bout_list to data frame
        bout_df = pd.DataFrame(bout_list)

        #if a max_first_index (from bout alignment) is provided, rows are sorted according
        #to the values in positions at and after `max_first_index` first. 
        if not (max_first_index is None): 
            sort_column_order = list(bout_df.columns[bout_df.columns >= max_first_index]) + list(bout_df.columns[bout_df.columns < max_first_index])

        #if no max_first_index is provided, rows are sorted alphanumerically starting with the first column
        else: 
            sort_column_order = list(bout_df.columns)
        
        #sort the rows based on the specified order of column importance
        bout_df = bout_df.sort_values(by = sort_column_order, ascending = True)

        #convert dataframe back to list of lists for consistency
        bout_list = bout_df.values.tolist()

        return bout_list
        

    def plot_syntax_raster(self, syntax_raster_df, figsize = (10, 10), title = None, 
                           palette = 'husl'):
        """
        Plots a syntax_raster_df dataframe. 

        Parameters
        ----------
        syntax_raster_df: Pandas DataFrame
            Dataframe where each row reflects a song bout (a sequence of syllables flanked by 
            file boundaries or long silent gaps), and each cell contains the label of the song 
            syllable produced at that index in the song bout, based on `self.syll_df.labels`.
            This is returned by `self.make_syntax_raster()`. 

        figsize: tuple, optional
            Tuple specifying dimensions of output figure. The default is (10, 10)

        title: String, optional
            Title of the output figure. The default is None, which will result in a figure
            without a title.
        
        palette: string or sequence, optional
            String corresponding to the name of a seaborn palette, matplotlib colormap
            or sequence of colors in any format matplotlib accepts. See `seaborn.color_palette()` 
            documentation for more information. The default is 'husl'. 

        Returns
        -------
        None
        
        """
        #get the number of unique syllable types 
        n = len(self.unique_labels)

        #specify a color map according to the number of syllable types
        cmap = sns.color_palette(palette, n)

        #create dictionary to convert characters to ints for plotting if bout_df contains 
        # character syllable labels
        if type(self.unique_labels[0]) != 'int':
            keys = self.unique_labels
            values = range(n)

            to_int_dict = dict(zip(keys, values))

            syntax_raster_df = syntax_raster_df.replace(to_int_dict)

        #plot syntax raster
        fig, ax = plt.subplots(figsize = figsize)
        ax = sns.heatmap(syntax_raster_df, cmap = cmap, linewidth = 0.5, mask = (syntax_raster_df.isna()))

        #add syllable labels to colorbar
        colorbar = ax.collections[0].colorbar
        r = colorbar.vmax - colorbar.vmin
        colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
        colorbar.set_ticklabels(list(to_int_dict.keys()))

        #hide x and y axis lines and ticks
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        #add title
        plt.title(title)

        plt.show()

    def _get_sylls_in_short_bouts(self, syll_df, max_short_bout_len = 2):
        """
        Creates a list of all syllable labels as they occur in bouts of duration 
        `max_short_bout_len` or shorter. This list can then be used to count the 
        total number of times a syllable type is produced in a short bout, which 
        can be useful for identifying calls. 

        Parameters
        ----------
        syll_df: Pandas DataFrame
            Pandas dataframe containing one row for every syllable to be analyzed
            from the subject bird. It must contain columns *onsets* and *offsets* 
            which contain the timestamp in seconds at which the syllable occurs 
            within a file, *files* which contains the name of the .wav file in 
            which the syllable is found, and *labels* which contains a categorical
            label for the syllable type. This syll_df must have calls (syllable 
            preceeded and followed by silent gaps or file boundaries) included. 

        max_short_bout_len: int, optional
            The maximum length of bout where the bout will be considered 'short' 
            and occurances of syllables within bouts of that length or shorter will 
            contribute to the count of syllables occuring in short bouts. 
            The default value is 2. 

        Returns
        -------
        all_sylls_in_short_bouts: list
            List of all syllable labels as they occur in bouts of duration 
            `max_short_bout_len` or shorter. This can then be used to count
            the total number of times a syllable type is produced in a short bout. 
        """

        #check whether calls have been dropped from syll_df
        #if so, use self.syll_df_with_calls instead
        if self.calls_dropped == True:
            syll_df = self.syll_df_with_calls
        else: 
            syll_df = self.syll_df
        
        #convert sequence of syllables to list of bouts
        bouts_list = self._syllables_to_bouts(syll_df.labels.to_list())

        #identify all bouts with <= max_short_bout_len syllables
        short_bouts = [bout for bout in bouts_list if len(bout) <= max_short_bout_len]

        #create list of all syllables in short_bouts
        all_sylls_in_short_bouts = [syllable for bout in short_bouts for syllable in bout]

        return all_sylls_in_short_bouts

    def get_prop_sylls_in_short_bouts(self, max_short_bout_len = 2):
        """
        Calculates the proportion of occurances of each syllable type in self.syll_df 
        occur in a bout with length equal to or shorter than `max_short_bout_len`. 
        This can be useful for identifying which syllable types reflect calls. 

        Parameters
        ----------
        max_short_bout_len: int, optional
            The maximum length of bout where the bout will be considered 'short' 
            and occurances of syllables within bouts of that length or shorter will 
            contribute to the count of syllables occuring in short bouts. 
            The default value is 2. 

        Returns
        -------
        all_syll_counts_df: Pandas DataFrame
            Dataframe with columns 'syllable', 'full_count', 'short_bout_count', 
            'Bird_ID' and 'prop_short_bout', where 'syllable' contains the label of 
            a syllable type in self.syll_df, 'full_count' contains the total number 
            of times that syllable occurs in self.syll_df, 'short_bout_count' contains
            the total number of times that syllable occurs in a bout of length 
            max_short_bout_len or shorter, and 'prop_short_bout' contains the proportion
            of all occurances of the syllable in short bouts. This proportion can be 
            useful for identifying which syllable types represent calls.  

        """
        #check whether calls have been dropped from self.syll_df
        #if so, use self.syll_df_with_calls instead
        if self.calls_dropped == True:
            syll_df = self.syll_df_with_calls
        else: 
            syll_df = self.syll_df
        
        #get list of occurances of syllables in short bouts
        all_sylls_in_short_bouts = self._get_sylls_in_short_bouts(syll_df, max_short_bout_len= max_short_bout_len)

        #make dataframe with counts of occurance of each syllable overall and in short bouts

        #convert all_sylls_in_short_bouts from list to Series
        all_sylls_in_short_bouts = pd.Series(all_sylls_in_short_bouts, dtype = 'category')
        #count number of occurances of each unique syllable type in short bouts
        short_bouts_counts = all_sylls_in_short_bouts.value_counts()
        #count number of occurances of each unique syllable type overall 
        all_sylls_counts = syll_df.labels.value_counts()
        #create dataframe with syllable count information
        all_syll_counts_df = pd.DataFrame({'full_count' : all_sylls_counts, 
                                           'short_bout_count' : short_bouts_counts, 
                                           'Bird_ID' : self.Bird_ID})
        #include syllable label as column, not index
        all_syll_counts_df = all_syll_counts_df.reset_index()
        all_syll_counts_df = all_syll_counts_df.rename(columns = {'index' : 'syllable'})
        #fill na values with zeros (na appears when the syllable wasn't detected in any short bouts, 
        # so the count should be reported as 0).
        all_syll_counts_df = all_syll_counts_df.fillna(value = 0)

        #add column to df with the proportion of all instances of the syllable which occur in short bouts
        all_syll_counts_df['prop_short_bout'] = all_syll_counts_df.short_bout_count / all_syll_counts_df.full_count

        #drop meaningless 'syllable' types
        all_syll_counts_df = all_syll_counts_df[~all_syll_counts_df['syllable'].isin(['file_start', 'silent_gap', 'file_end'])]

        return all_syll_counts_df

    def _check_IN_transition_pattern(self, candidate_IN):
        """
        Determines whether `candidate_IN` is likely to be an intro note. Will return True
        if candidate_IN is among the syllables most commonly transitioned to from silence 
        and makes one dominant transition to another syllable type other than itself which
        is not silence. Otherwise it returns False. 

        Parameters
        ----------
        candidate_IN: string, int or char
            The label of a syllable type in self.syll_df.labels.

        Returns
        -------
        bool
            True if the syllable meets syntactic criteria for an intro note. False if the 
            syllable does not meet the criteria. 

        """
        #check whether candidate_IN is among the syllables most frequently transitioned to from silence. 
        #if not, return False, as it is unlikely to reflect and intro note. 
        max_transition_from_silence = self.trans_mat_prob.loc['silent_gap'].max()
        acceptable_range_for_transitions_from_silence = max_transition_from_silence - 0.05

        if self.trans_mat_prob.loc['silent_gap', candidate_IN] < acceptable_range_for_transitions_from_silence:
            return False
        
        else:
            #get probability that syllable transitions to itself
            prob_trans_to_self = self.trans_mat_prob.loc[candidate_IN, candidate_IN]
            #calculate probability necessary for syllable to have one other dominant transition
            dominant_other_transition_cutoff = (1 - prob_trans_to_self) / 2

            #get boolean mask of transitions that exceed dominant transition cutoff
            transitions_meeting_cutoff_bool = (self.trans_mat_prob.loc[candidate_IN] >= dominant_other_transition_cutoff)

            #ignore transition from candidate_IN to itself when checking for a single other dominant transition
            transitions_meeting_cutoff_bool[candidate_IN] = False

            #check that there is exactly one transition that meets the criteria
            if transitions_meeting_cutoff_bool.sum() == 1:
                #check whether the one transition is to 'silent_gap'
                syll_transitioned_to = transitions_meeting_cutoff_bool.index[transitions_meeting_cutoff_bool].to_list()
                if syll_transitioned_to == ['silent_gap']:
                    return False
                else:
                    return True
            else: 
                return False

    def get_intro_notes_df(self):
        """
        Determines whether each syllable type in self.syll_df is likely to be an intro note. 
        A syllable is considered a possible intro note if: 
            1) the syllable is among the most common syllables transitioned to from silence
            2) AND the syllable makes done dominant transition other than to itself to a 
                syllable that is not silence. 
        
        Returns
        -------
        all_intro_notes: Pandas DataFrame
            Dataframe with columns 'syllable', 'Bird_ID' and 'intro_note' where each row 
            corresponds to a syllable type in self.syll_df, and 'intro_note' contains a 
            boolean value reflecting whether or not the syllable meets criteria to be 
            considered an intro note. 

        """
        #initialize empty dataframe to store intro note status per syllable
        all_intro_notes = pd.DataFrame()
        #loop over each syllable type
        for syllable in self.unique_labels:
            #determine whether syllable could be an intro note
            is_intro_note = self._check_IN_transition_pattern(syllable)

            #record intro note status in dataframe
            curr_df = pd.DataFrame({'syllable': [syllable], 
                                    'Bird_ID': [self.Bird_ID], 
                                    'intro_note': is_intro_note})
            #append df for current syllable to df with all intro notes
            all_intro_notes = pd.concat([all_intro_notes, curr_df])

        return all_intro_notes
        
    def save_syntax_data(self, output_directory):
        """
        Saves a copy of `.syll_df` as a .csv file in the output directory. Also saves 
        CSVs of the transition matrices if they exist, and a syntax analysis metadata.csv
        file with information on the processes used to modify syll_df and create the transition
        matrices

        Parameters
        ----------
        output_directory: string
            Path to a folder in which to save `[Bird_ID]_syll_df.csv`, 
            `[Bird_ID]_syntax_analysis_metadata.csv`, and `[Bird_ID]_trans_mat.csv` and 
            `[Bird_ID]_trans_mat_prob.csv`, if they exist. 

        Returns
        -------
        syntax_analysis_metadata: Pandas DataFrame
            Dataframe containing information about the package version and processing steps 
            used in creating the versions of .syll_df and transition matrices that were saved 
            by the function. 
         
        """
        #save the syll_df
        syll_df_out_path = output_directory + self.Bird_ID + '_syll_df.csv'
        self.syll_df.to_csv(syll_df_out_path)

        #check whether transition matrices exist.
        #if the transition matrices do exist: 
        if hasattr(self, 'trans_mat'):
            #set status for metadata
            trans_mat_status = True
            #save trans_mat
            trans_mat_out_path = output_directory + self.Bird_ID + '_trans_mat.csv'
            self.trans_mat.to_csv(trans_mat_out_path)
            #save trans_mat_prob
            trans_mat_prob_out_path = output_directory + self.Bird_ID + '_trans_mat_prob.csv'
            self.trans_mat_prob.to_csv(trans_mat_prob_out_path)

        else:    
            #set status for metadata
            trans_mat_status = False
            trans_mat_out_path = None
            trans_mat_prob_out_path = None

        #create metadata dataframe
        syntax_analysis_metadata = pd.DataFrame({'Date' : [datetime.date.today().strftime('%Y-%m-%d')], 
                                                 'avn_version' : [avn.__version__],
                                                 'syll_df_path' : syll_df_out_path,
                                                 'file_boundaries_added' : self.file_bounds_added,
                                                 'gaps_added' : self.gaps_added, 
                                                 'min_gap' : self.min_gap, 
                                                 'calls_dropped' : self.calls_dropped, 
                                                 'transition_matrices_saved' : trans_mat_status, 
                                                 'trans_mat_path' : trans_mat_out_path, 
                                                 'trans_mat_prob_path' : trans_mat_out_path })
        #save metadata dataframe
        syntax_analysis_metadata.to_csv(output_directory + self.Bird_ID + '_syntax_analysis_metadata.csv')

        return syntax_analysis_metadata






class Utils:
    """
    Contains syntax analysis utilities
    """

    def __init__():
        pass

    def plot_transition_matrix_all_birds(Bird_IDs, syll_df_folder_path, syll_df_file_name_suffix, song_folder_path,
                                         min_gap = 0.2, calc_entropy_rate = True, 
                                         label_column_name = None, trans_mat_version = 'prob', 
                                         figsize = (10,8)):
        """
        Plots the transition matrices of all birds in Bird_IDs 

        Parameters
        ----------
        Bird_IDs: list of strings
            List of Bird_IDs (as strings) for which the transition matrix should be plotted. 
        
        syll_df_folder_path: string
            Path to a folder containing a syll_df for each bird in Bird_ID. 
            The syll_df must be a dataframe with one row for every syllable 
            to be analyzed from the subject bird. It must contain columns *onsets* and
            *offsets* which contain the timestamp in seconds at which the syllable occurs 
            within a file, *files* which contains the name of the .wav file in 
            which the syllable is found, and *labels* which contains a categorical
            label for the syllable type. These can be generated through manual song
            annotation, or automated labeling methods. The syll_df files must be .csv
            files named Bird_ID_`syll_df_file_name_suffix`. 

        syll_df_file_name_suffix: string
            String that specifies the name of the file containing syll_df for each Bird_ID. 
            For example, if syll_df files are named 'Bird_ID_syll_df.csv', `syll_df_file_name_suffix`
            should be '_syll_df.csv'. 

        song_folder_path: string
            Path to a folder containing subfolders named according to the Bird_IDs, where each 
            subfolder contains the complete set of .wav files used to generate the syll_df loaded 
            from `syll_df_folder_path`. 

        min_gap: float, optional
            Minimum duration in seconds for a gap between syllables to be 
            considered syntactically relevant. This value should be selected 
            such that gaps between syllables in a bout are shorter than min_gap, 
            but gaps between bouts are longer than min_gap. The default is 0.2.

        calc_entropy_rate: bool, optional
            Determines whether entropy rate is calculated for each bird. If True, 
            entropy rate will be calculated and reported in the title of the transition 
            matrix plot for each bird. The default is True.
        
        label_column_name: string, optional
            If the column of the syll_df containing syllable labels is not called 'labels', 
            the name of that column should be specified here as a string. If no value is 
            provided, an existing column called 'labels' in syll_df will be used as syllable labels.

        trans_mat_version: 'prob' or 'count', optional
            Specifies whether to plot transition probabilities in the transition matrix
            or counts of transitions in the dataset. The default value is 'prob' which 
            results in the plotting of transition probabilities between syllables. 

        figsize: tuple, optional
            Tuple which sets the dimensions of each output transition matrix plot. 

        Returns
        -------
        None

        """
        #loop over each Bird_ID in Bird_IDs
        for Bird_ID in Bird_IDs:

            #load syll_df from .csv file
            syll_df = pd.read_csv(syll_df_folder_path + Bird_ID + syll_df_file_name_suffix)

            #if label_column_name is provided, use data from that column as primary syllable 
            #label going forward. 
            if not (label_column_name == None):
                syll_df['labels'] = syll_df[label_column_name]

            #create syntax data object
            syntax_data = SyntaxData(Bird_ID, syll_df)

            #add file boundaries
            syntax_data.add_file_bounds(song_folder_path)

            #add gaps
            syntax_data.add_gaps(min_gap=min_gap)

            #drop calls
            syntax_data.drop_calls()

            #make transition matrix
            syntax_data.make_transition_matrix()

            #calculate entropy rate if calc_entropy_rate == True
            if calc_entropy_rate == True:
                entropy_rate = syntax_data.get_entropy_rate()
                plot_title = Bird_ID + " Entropy Rate = " + str(entropy_rate)

            else: 
                plot_title = Bird_ID

            #plot transition matrix
            if trans_mat_version == 'prob':
                plt.figure(figsize = figsize)
                sns.heatmap(syntax_data.trans_mat_prob, annot = True, fmt = '0.2f', vmin = 0, vmax = 1)
                plt.title(plot_title)
                plt.show()

            if trans_mat_version == 'count':
                plt.figure(figsize = figsize)
                sns.heatmap(syntax_data.trans_mat, annot = True, fmt = '0.0f')
                plt.title(plot_title)
                plt.show()

    def calc_entropy_rate_all_birds(Bird_IDs, syll_df_folder_path, syll_df_file_name_suffix, song_folder_path,
                                    min_gap = 0.2, label_column_name = None):
        """
        Creates a dataframe with the syntax entropy rate of each bird in Bird_IDs. 

        Parameters
        ----------
        Bird_IDs: list of strings
            List of Bird_IDs (as strings) for which the transition matrix should be plotted. 
        
        syll_df_folder_path: string
            Path to a folder containing a syll_df for each bird in Bird_IDs. 
            The syll_df must be a dataframe with one row for every syllable 
            to be analyzed from the subject bird. It must contain columns *onsets* and
            *offsets* which contain the timestamp in seconds at which the syllable occurs 
            within a file, *files* which contains the name of the .wav file in 
            which the syllable is found, and *labels* which contains a categorical
            label for the syllable type. These can be generated through manual song
            annotation, or automated labeling methods. The syll_df files must be .csv
            files named Bird_ID_`syll_df_file_name_suffix`. 

        syll_df_file_name_suffix: string
            String that specifies the name of the file containing syll_df for each Bird_ID. 
            For example, if syll_df files are named 'Bird_ID_syll_df.csv', `syll_df_file_name_suffix`
            should be '_syll_df.csv'. 

        song_folder_path: string
            Path to a folder containing subfolders named according to the Bird_IDs, where each 
            subfolder contains the complete set of .wav files used to generate the syll_df loaded 
            from `syll_df_folder_path`. 

        min_gap: float, optional
            Minimum duration in seconds for a gap between syllables to be 
            considered syntactically relevant. This value should be selected 
            such that gaps between syllables in a bout are shorter than min_gap, 
            but gaps between bouts are longer than min_gap. The default is 0.2.

        label_column_name: string, optional
            If the column of the syll_df containing syllable labels is not called 'labels', 
            the name of that column should be specified here as a string. If no value is 
            provided, an existing column called 'labels' in syll_df will be used as syllable labels.

        Returns
        -------
        all_entropy_rates: Pandas DataFrame
            Dataframe with columns 'Bird_ID', 'entropy_rate', 'num_unique_syll_types' and 
            'entropy_rate_norm'. 'entropy_rate' contains the raw syntax entropy rate, and 
            'entropy_rate_norm' contains an entopy rate value that is normalized to account
            for the number of unique syllable types. 
        
        """
        #initialize empty dataframe to store entropy rates from all birds
        all_entropy_rates = pd.DataFrame()

        #loop over each Bird_ID in Bird_IDs
        for Bird_ID in Bird_IDs:

            #load syll_df from .csv file
            syll_df = pd.read_csv(syll_df_folder_path + Bird_ID + syll_df_file_name_suffix)

            #if label_column_name is provided, use data from that column as primary syllable 
            #label going forward. 
            if not (label_column_name == None):
                syll_df['labels'] = syll_df[label_column_name]

            #create syntax data object
            syntax_data = SyntaxData(Bird_ID, syll_df)

            #add file boundaries
            syntax_data.add_file_bounds(song_folder_path)

            #add gaps
            syntax_data.add_gaps(min_gap=min_gap)

            #drop calls
            syntax_data.drop_calls()

            #make transition matrix
            syntax_data.make_transition_matrix()

            #calculate entropy rate
            entropy_rate = syntax_data.get_entropy_rate()

            #get the number of unique syllable types - This can be used to normalize
            #the entropy rate to the number of syllable classes. 
            num_syll_classes = len(syntax_data.unique_labels) + 1 #plus 1 because 'silent_gap' is also a relevant 
                                                                  #syllable state for entropy rate calculation

            #package entropy rate into dataframe
            curr_df = pd.DataFrame({'Bird_ID' : [Bird_ID], 
                                    'entropy_rate': [entropy_rate], 
                                    'num_unique_syll_types': [num_syll_classes], 
                                    'entropy_rate_norm': [entropy_rate / np.log2(num_syll_classes)]})

            #append current df to df with entropy rate from all birds
            all_entropy_rates = pd.concat([all_entropy_rates, curr_df])

        return all_entropy_rates
                        

    def plot_syntax_raster_all_birds(Bird_IDs, syll_df_folder_path, syll_df_file_name_suffix, song_folder_path,
                                    min_gap = 0.2, label_column_name = None, figsize = (10,10), sort_bouts = True, 
                                    calc_entropy_rate = True):
        """
        Plots the syntax raster plot for each bird in Bird_IDs. 

        Parameters
        ----------
        Bird_IDs: list of strings
            List of Bird_IDs (as strings) for which the transition matrix should be plotted. 
        
        syll_df_folder_path: string
            Path to a folder containing a syll_df for each bird in Bird_IDs. 
            The syll_df must be a dataframe with one row for every syllable 
            to be analyzed from the subject bird. It must contain columns *onsets* and
            *offsets* which contain the timestamp in seconds at which the syllable occurs 
            within a file, *files* which contains the name of the .wav file in 
            which the syllable is found, and *labels* which contains a categorical
            label for the syllable type. These can be generated through manual song
            annotation, or automated labeling methods. The syll_df files must be .csv
            files named Bird_ID_`syll_df_file_name_suffix`. 

        syll_df_file_name_suffix: string
            String that specifies the name of the file containing syll_df for each Bird_ID. 
            For example, if syll_df files are named 'Bird_ID_syll_df.csv', `syll_df_file_name_suffix`
            should be '_syll_df.csv'. 

        song_folder_path: string
            Path to a folder containing subfolders named according to the Bird_IDs, where each 
            subfolder contains the complete set of .wav files used to generate the syll_df loaded 
            from `syll_df_folder_path`. 

        min_gap: float, optional
            Minimum duration in seconds for a gap between syllables to be 
            considered syntactically relevant. This value should be selected 
            such that gaps between syllables in a bout are shorter than min_gap, 
            but gaps between bouts are longer than min_gap. The default is 0.2.

        label_column_name: string, optional
            If the column of the syll_df containing syllable labels is not called 'labels', 
            the name of that column should be specified here as a string. If no value is 
            provided, an existing column called 'labels' in syll_df will be used as syllable labels.

        figsize: tuple, optional
            Tuple to specify dimensions of each output syntax raster plot. The default is (10,10).

        sort_bouts: bool,  optional
            If True, bouts will be sorted such that bouts with more similar sequences will occupy
            sequential rows in the plot. This can make it easier to detect syntax patterns
            agnostic to the order in which bouts were produced. If False, the order of bouts in 
            `syntax_raster_df` will be the order in which the bouts occur in `self.syll_df`. 
            The default is True.

        calc_entropy_rate: bool, optional
            Determines whether entropy rate is calculated for each bird. If True, 
            entropy rate will be calculated and reported in the title of the syntax raster plot for each bird.
            The default is True.

        Returns
        -------
        None

        """
            
        #loop over each Bird_ID in Bird_IDs
        for Bird_ID in Bird_IDs:

            #load syll_df from .csv file
            syll_df = pd.read_csv(syll_df_folder_path + Bird_ID + syll_df_file_name_suffix)

            #if label_column_name is provided, use data from that column as primary syllable 
            #label going forward. 
            if not (label_column_name == None):
                syll_df['labels'] = syll_df[label_column_name]

            #create syntax data object
            syntax_data = SyntaxData(Bird_ID, syll_df)

            #add file boundaries
            syntax_data.add_file_bounds(song_folder_path)

            #add gaps
            syntax_data.add_gaps(min_gap=min_gap)

            #drop calls
            syntax_data.drop_calls()

            #make syntax raster dataframe
            syntax_raster_df = syntax_data.make_syntax_raster(sort_bouts = sort_bouts)

            #if calc_entropy_rate == True, calculate the entropy rate and display it in the figure title
            if calc_entropy_rate == True:
                syntax_data.make_transition_matrix()
                entropy_rate = syntax_data.get_entropy_rate()
                title = Bird_ID + " Entropy Rate = " + str(entropy_rate)
            
            else:
                title = Bird_ID

            #plot syntax raster
            syntax_data.plot_syntax_raster(syntax_raster_df, figsize = figsize, title = title)

    def merge_per_syll_stats(single_rep_stats, short_bout_counts, intro_notes_df):
        """
        Merge 3 dataframes containing syntax related measures per syllable type into a single
        dataframe with all per syllable syntax stats. 

        Parameters
        ----------
        single_rep_stats: Pandas DataFrame
            Dataframe with columns 'Bird_ID' and 'syllable', as well as other columns 
            with summary statistics, which contains one row per unique syllable type 
            in the bird's repertoire. This could be the single_rep_stats
            dataframe returned by `.get_single_repetition_stats()`. 

        short_bout_counts: Pandas DataFrame
            Dataframe with columns 'Bird_ID' and 'syllable', as well as other columns 
            with summary statistics, which contains one row per unique syllable type 
            in the bird's repertoire. This could be the 'short_bout_counts' dataframe
            returned by `.get_prop_sylls_in_short_bouts()`. 

        intro_notes_df: Pandas DataFrame
            Dataframe with columns 'Bird_ID' and 'syllable', as well as other columns 
            with summary statistics, which contains one row per unique syllable type 
            in the bird's repertoirer. This could be the 'intro_notes_df' dataframe 
            returned by `.get_intro_notes_df()`.

        Returns
        -------
        syllable_syntax_stats: Pandas DataFrame
            DataFrame resulting from merge of the 3 input dataframes on columns 'Bird_ID'
            and 'syllable'. 
        
        """
        #merge single_rep_stats and short_bout_counts
        syllable_syntax_stats = single_rep_stats.merge(short_bout_counts, on = ['Bird_ID', 'syllable'])

        #merge with intro note df
        syllable_syntax_stats = syllable_syntax_stats.merge(intro_notes_df, on = ['Bird_ID', 'syllable'])

        return syllable_syntax_stats

    def get_syll_stats_all_birds(Bird_IDs, syll_df_folder_path, syll_df_file_name_suffix, song_folder_path,
                                 min_gap = 0.2, label_column_name = None, max_short_bout_len = 2):
        """
        Compile all per-syllable syntax statistics from all birds in Bird_IDs into a single
        dataframe. This dataframe can then be used to detect syllable with abnormal repetition 
        patterns. 

        Parameters
        ----------
        Bird_IDs: list of strings
            List of Bird_IDs (as strings) for which the transition matrix should be plotted. 
        
        syll_df_folder_path: string
            Path to a folder containing a syll_df for each bird in Bird_IDs. 
            The syll_df must be a dataframe with one row for every syllable 
            to be analyzed from the subject bird. It must contain columns *onsets* and
            *offsets* which contain the timestamp in seconds at which the syllable occurs 
            within a file, *files* which contains the name of the .wav file in 
            which the syllable is found, and *labels* which contains a categorical
            label for the syllable type. These can be generated through manual song
            annotation, or automated labeling methods. The syll_df files must be .csv
            files named Bird_ID_`syll_df_file_name_suffix`. 

        syll_df_file_name_suffix: string
            String that specifies the name of the file containing syll_df for each Bird_ID. 
            For example, if syll_df files are named 'Bird_ID_syll_df.csv', `syll_df_file_name_suffix`
            should be '_syll_df.csv'. 

        song_folder_path: string
            Path to a folder containing subfolders named according to the Bird_IDs, where each 
            subfolder contains the complete set of .wav files used to generate the syll_df loaded 
            from `syll_df_folder_path`. 

        min_gap: float, optional
            Minimum duration in seconds for a gap between syllables to be 
            considered syntactically relevant. This value should be selected 
            such that gaps between syllables in a bout are shorter than min_gap, 
            but gaps between bouts are longer than min_gap. The default is 0.2.

        label_column_name: string, optional
            If the column of the syll_df containing syllable labels is not called 'labels', 
            the name of that column should be specified here as a string. If no value is 
            provided, an existing column called 'labels' in syll_df will be used as syllable labels.

        max_short_bout_len: int, optional
            The maximum length of bout where the bout will be considered 'short' 
            and occurances of syllables within bouts of that length or shorter will 
            contribute to the count of syllables occuring in short bouts. This is 
            used to identify calls. The default value is 2.

        Returns
        --------
        syll_stats_all_birds: Pandas DataFrame
            Dataframe with one row for each unique syllable type produced by each 
            bird in Bird_IDs containing information about the repetition and 
            syntax patterns of each syllable. This can be used for detecting 
            abnormal syllable types with `Utils.identify_abnormal_syllables()`. 

        
        """
        #initialize empty dataframe to store per syll type stats from all birds
        syll_stats_all_birds = pd.DataFrame()
        
        #loop over each Bird_ID in Bird_IDs
        for Bird_ID in Bird_IDs:

            #load syll_df from .csv file
            syll_df = pd.read_csv(syll_df_folder_path + Bird_ID + syll_df_file_name_suffix)

            #if label_column_name is provided, use data from that column as primary syllable 
            #label going forward. 
            if not (label_column_name == None):
                syll_df['labels'] = syll_df[label_column_name]

            #create syntax data object
            syntax_data = SyntaxData(Bird_ID, syll_df)

            #add file boundaries
            syntax_data.add_file_bounds(song_folder_path)

            #add gaps
            syntax_data.add_gaps(min_gap=min_gap)

            #drop calls
            syntax_data.drop_calls()

            #make transition matrix
            syntax_data.make_transition_matrix()

            #get repetition stats per syllable type
            __, single_rep_stats = syntax_data.get_single_repetition_stats()

            #get proportion of syllables produced in bouts with length <= max_short_bout_len syllables
            short_bout_counts = syntax_data.get_prop_sylls_in_short_bouts(max_short_bout_len)

            #check whether syllable has syntax properties of an intro note
            intro_notes_df = syntax_data.get_intro_notes_df()

            #merge 3 datasets with per syllable type measures
            per_syll_stats = Utils.merge_per_syll_stats(single_rep_stats, short_bout_counts, intro_notes_df)

            #append dataframe with stats from current bird to df with all birds. 
            syll_stats_all_birds = pd.concat([syll_stats_all_birds, per_syll_stats])

        return syll_stats_all_birds

    def identify_abnormal_syllables(syll_stats_all_birds, std_cutoff = 2, exclude_calls = True, exclude_intro_notes = True, 
                                    syll_labels_to_exclude = [-1], prop_short_bout_cutoff = 0.5):
        """
        Identifies syllables that are over `std_cutoff` standard deviations from the mean 
        in terms of mean_repetition_length or CV_repetition_length, and returns a version of
        syll_stats_all_birds with a new column 'abnormal_repetition' containing a boolean
        to indicate whether that syllable exhibits unusually high repetition or repetition
        variability. 

        Parameters
        ----------

        syll_stats_all_birds: Pandas DataFrame
            Dataframe with one row for each unique syllable type produced by each 
            bird in Bird_IDs containing information about the repetition and 
            syntax patterns of each syllable.

        std_cutoff: float, optional
            The number of standard deviations from the mean a syllable feature must be 
            for that syllable to be identified as 'abnormal'. The default value is 2. 

        exclude_calls: bool, optional
            If True, syllables which occur in short bouts  > `prop_short_bout_cutoff` 
            proprotion of the time will be considered calls, and not be considered when 
            calculating the mean and std used to identify abnormal syllable types. 
            These calls will also cannot be identified as 'abnormal'. 
            If False, syllable occuring in short bouts at high rates will be treated like 
            standard syllables. The default value is True. 
        
        exclude_intro_notes: bool, optional
            If True, syllables with `intro_note`  == True will not be considered when 
            calculating the mean and std used to identify abnormal syllable types. 
            These intro notes also cannot be identified as 'abnormal'. 
            If False, intro notes will be treated like standard syllables. 
            The default value is True. 

        syll_labels_to_exclude: list, optional
            List of syllable labels that should not be  considered when calculating the 
            mean and std used to identify abnormal syllables. For example, if syllables 
            are labeled automatically with HDBSCAN, the label '-1' doesn't reflect a 
            relevant grouping of syllables, and thus shouldn't contribute to population 
            statistics about syllable repetition patterns. 
            The default value is [-1]. 

        prop_short_bout_cutoff: float between 0 and 1, optional
            If exclude_calls == True, syllables with which occur in short bouts with a 
            proportion greater than this value will be considered calls and be excluded 
            from analysis of abnormal syllables. 

        Returns
        -------
        syll_stats_all_birds: Pandas DataFrame
            Copy of input `syll_stats_all_birds` dataframe, with will a column called 
            'abnormal_repetition' added, which contains a boolean value indicating 
            whether that syllable has a mean_repetition_length or CV_repetition_length
            over `std_cutoff` standard deviations from the mean. 
        
        """
        #drop all rows with syllable labels in syll_labels_to_exclude
        syll_stats_filtered = syll_stats_all_birds[~ syll_stats_all_birds.syllable.isin(syll_labels_to_exclude)]

        #if exclude_intro_notes == True, drop rows that likely reflect intro notes
        if exclude_intro_notes == True :
            syll_stats_filtered = syll_stats_filtered[syll_stats_filtered.intro_note == False]

        #if exclude_calls == True, drop rows where prop_short_bout > prop_short_bout_cutoff
        if exclude_calls == True:
            syll_stats_filtered = syll_stats_filtered[syll_stats_filtered.prop_short_bout <= prop_short_bout_cutoff]

        #calculate the cutoff for mean_repetition_length to be considered abnormal
        mean_mean_rep_length = syll_stats_filtered.mean_repetition_length.mean()
        std_mean_rep_length = syll_stats_filtered.mean_repetition_length.std()
        mean_rep_length_cutoff = mean_mean_rep_length + std_cutoff * std_mean_rep_length

        #calculate the cutoff for CV_repetition_length to be considered abnormal
        mean_CV_rep_length = syll_stats_filtered.CV_repetition_length.mean()
        std_CV_rep_length = syll_stats_filtered.CV_repetition_length.std()
        CV_rep_length_cutoff = mean_CV_rep_length + std_cutoff * std_mean_rep_length

        #create new column to indicate whether syllable is abnormal
        syll_stats_all_birds['abnormal_repetition'] = False

        #set abdnormal_repetition = True for all syllables that exceed the mean or CV rep length cutoffs
        high_mean_mask = syll_stats_all_birds.mean_repetition_length >= mean_rep_length_cutoff
        high_CV_mask = syll_stats_all_birds.CV_repetition_length >= CV_rep_length_cutoff
        syll_stats_all_birds.loc[high_mean_mask | high_CV_mask, 'abnormal_repetition'] = True

        #set abnormal_repetition = False for all syllables that reflect intro notes or calls
        if exclude_intro_notes == True : 
            syll_stats_all_birds.loc[syll_stats_all_birds.intro_note == True, 'abnormal_repetition'] = False

        if exclude_calls == True: 
            syll_stats_all_birds.loc[syll_stats_all_birds.prop_short_bout > prop_short_bout_cutoff, 'abnormal_repetition'] = False

        return syll_stats_all_birds



        


