# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:39:10 2021

@author: Therese
"""
import avn.dataloading as dataloading 
#import dataloading
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

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
    return fig

def plot_syntax_raster(syntax_data, syntax_raster_df, figsize = (10, 10), title = None, 
                           palette = 'husl'):
    """
    Plots a syntax_raster_df dataframe. 

    Parameters
    ----------
    syntax_data: avn.syntax.SyntaxData object
        An instance of avn.syntax.SyntaxData on which `.make_synta_raster()` was called 
        to generate `syntax_raster_df`. 

    syntax_raster_df: Pandas DataFrame
        Dataframe where each row reflects a song bout (a sequence of syllables flanked by 
        file boundaries or long silent gaps), and each cell contains the label of the song 
        syllable produced at that index in the song bout, based on `syntax_data.syll_df.labels`.
        This is returned by `.make_syntax_raster()` called on an instance of a 
        `syntax.SyntaxData` object. 

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
    n = len(syntax_data.unique_labels)

    #specify a color map according to the number of syllable types
    cmap = sns.color_palette(palette, n)

    #create dictionary to convert characters to ints for plotting if bout_df contains 
    # character syllable labels
    if type(syntax_data.unique_labels[0]) != 'int':
        keys = syntax_data.unique_labels
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

    #plt.show()
    return fig

def plot_spectrogram_with_labels(syll_df, song_folder_path, Bird_ID, song_file = None, song_file_index = None, figsize = (80, 10), 
                                 cmap = 'tab20',  add_legend = True, fontsize = 24):
    """
    Plots the sectrogram of a specified file with syllable labels indicated through colored bars overlaid on spectrogram. 

    Parameters
    ----------
    syll_df: Pandas DataFrame
        pandas dataframe containing one row for every syllable to be analyzed
        from the subject bird. It must contain columns *onsets* and *offsets* 
        which contain the timestamp in seconds at which the syllable occurs 
        within a file, *files* which contains the name of the .wav file in 
        which the syllable is found, and *labels* which contains a categorical
        label for the syllable type. These can be generated through manual song
        annotation, or automated labeling methods. 

    song_folder_path: string
        Path to folder containing a subfolder called `Bird_ID`, which contains 
        .wav files of songs in syll_df. Should end with '/'.

    Bird_ID: string
        String containing a unique identifier for the subject bird.

    song_file: string, optional
        A value must be provided for `song_file` OR `song_file_index`, but not both. 
        String containing the name of a .wav file in `syll_df` and `song_folder_path/Bird_ID/` 
        to be plotted. 

    song_file_index: int >= 0, optional
        A value must be provided for `song_file` or `song_file_index`, but not both. 
        Denotes the index of the unique file in syll_df.files.unique() to be plotted. 

    figsize: tuple, optional
        Dimensions of figure to be plotted. The default is (80, 10). 

    cmap: matplotlib colormap, optional
        matplotlib color map for syllable labels. The colormap must contain more unique shades 
        than syllable label types. The default value is 'tab20'. 

    add_legend: boolean, optional
        If True, a legend mapping syllable labels to colors will be plotted over the spectrogram. 
        If False, no legend will be plotted. The default is True. 

    fontsize: float, optional
        The size of the font for the legend. The default is 24. 

    Returns
    -------
    None

    """
    #if both song file and song file index are prodivided, which file to plot if ambiguous so raise an error
    if (song_file != None) & (song_file_index != None):
        raise RuntimeError("Both `song_file` and `song_file_index` were provided. Please provide only one, otherwise the file to be plotted is ambiguous.")

    #set song file name according to song_file if it is provided
    if song_file != None: 
        song_file_name = song_file
    
    #set song file name according to song_file_index if it is provided
    if song_file_index != None: 
        song_file_name = syll_df.files.unique()[song_file_index]

    #if neither song file nor song file index are provided we will just plot the first file, but raise a warning
    if (song_file == None) & (song_file_index == None):
        song_file_name = syll_df.files.unique()[0]
        warnings.warn("No song_file or song_file_index were provided. The first file in syll_df will be plotted. To avoid ambiguity, please specify either song_file or song_file_index.")

    #set path to song file
    file_path = song_folder_path + Bird_ID + '/' + song_file_name

    #load song file
    song = dataloading.SongFile(file_path)
    song.bandpass_filter(500, 15000)

    #make spectrogram
    spectrogram = make_spectrogram(song)

    #plot spectrogram
    fig, ax = plt.subplots(figsize = figsize)
    plot_spectrogram(spectrogram, song.sample_rate, ax = ax)
    #set title
    ax.set_title(Bird_ID + "    " + song_file_name)

    #get y dimensions for adding syll labels
    ymin, ymax = ax.get_ylim()
    
    #create color dict for plotting syllable labels
    labels = syll_df.labels.unique()
    colors = plt.cm.get_cmap(cmap)(np.arange(len(labels)))#there must be more unique colors in cmap than label types
    color_dict = dict(zip(labels, colors))

    #loop over each syllable in syll_table in song_file:
    for ix, row in syll_df[syll_df.files == song_file_name].iterrows():

        #set color for syllable label
        color = color_dict[row.labels]

        #add patch to represent syllable label
        ax.add_patch(mpatches.Rectangle( [row.onsets, ymax - (ymax - ymin) / 10], 
                                         row.offsets -  row.onsets, 
                                         (ymax  - ymin) / 10,
                                         ec = 'none', 
                                         color = color))
    
    #if add_legend == True, add legend
    if add_legend == True: 
        markers  = [plt.Line2D([0, 0], [0, 0],  color = color, marker = 'o', linestyle = '') for color in color_dict.values()]
        ax.legend(markers, color_dict.keys(), numpoints = 1, facecolor = 'black', 
                  labelcolor = 'white', markerscale = 3, fontsize = fontsize, loc = 'upper right')
        
    return fig, ax, song_file_name
    
def plot_syll(song, onset, offset, padding = 0, figsize = (5,5), title = None):
    """
    Plots the spectrogram of a portion of a song spectrogram (generally a single syllable).

    Parameters
    ----------
    song: avn.dataloading.SongFile type object
        Instance of an avn.SongFile object with `.data`, `.sample_rate` and `.duration` 
        attributes.

    onset: float
        Start time in seconds of syllable to plot from `song`. 

    offset: float
        End time in seconds of syllable to plot from `song`. 
    
    padding: float, optional
        Time in seconds to pad before and after onset and offset times for plotting. 
        The default value is 0. 

    figsize: tuple, optional
        Dimensions of the figure to plot. The default is (5,5)

    title: string, optional 
        Title of the figure to plot. The default is None. 

    Returns
    -------
    None
    
    """
    syll_data, onset_correction, offset_correction = dataloading.Utils.select_syll(song, onset, offset, padding)

    song.data = syll_data

    spectrogram = make_spectrogram(song)

    fig, ax = plt.subplots(figsize = figsize)
    plot_spectrogram(spectrogram, song.sample_rate, ax = ax)

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()

    ax.set_title(title)

    plot_onset = xmin + padding + onset_correction
    plot_offset = xmax - padding + offset_correction

    ax.add_patch(mpatches.Rectangle([plot_onset, ymax - (ymax - ymin)/10], 
                                    plot_offset - plot_onset, 
                                    (ymax - ymin) / 10, 
                                    ec = 'none', 
                                    color = 'red'))

    
class Utils:
    """
    Containts plotting utilities. 
    """

    def __init__(): 
        pass
    
    def plot_syll_examples(syll_df, syll_label, song_folder_path, n_examples = 1, random_seed = 2021, padding = 0.25, figsize = (5,5)):

        """
        Plots `n_examples` examples of syllables with label `syll_label` from `syll_df`. 

        Parameters
        ----------

        syll_df: Pandas DataFrame
            pandas dataframe containing one row for every syllable to be analyzed
            from the subject bird. It must contain columns *onsets* and *offsets* 
            which contain the timestamp in seconds at which the syllable occurs 
            within a file, *files* which contains the name of the .wav file in 
            which the syllable is found, and *labels* which contains a categorical
            label for the syllable type. These can be generated through manual song
            annotation, or automated labeling methods.

        syll_label: int, float or string
            Syllable label class in `syll_df.labels` to plot. 

        song_folder_path: string
            Path to folder containing a subfolder called `Bird_ID`, which contains 
            .wav files of songs in syll_df. Should end with '/'.

        n_examples: int, optional
            The number of random examples of syllable `syll_label` to plot. 
            The default value is 1. 

        random_seed: int, optional
            Specifies the random state for selecting example syllables. 
            The default value is 2021. 

        padding: float, optional
            The amount of time in seconds before and after syllable onset and offset
            which should be included in the spectrogram plot. 
            The default value is 0.25.

        figsize: tuple, optional
            Tuple specifying the dimensions of the figure(s) to be plotted. 
            The default value is (5,5)


        """

        #select subset of syll_df where 'label' == syll_label
        syll_df = syll_df[syll_df.labels == syll_label] 

        #if there are fewer examples of syll_df than n_examples, raise a warning
        if syll_df.shape[0] < n_examples:
            warnings.warn(("There are fewer than " + str(n_examples) + " instances of syllable '" + str(syll_label) + "' in syll_df. All " + str(syll_df.shape[0]) + " instances will be plotted instead."))
            #update n_examples
            n_examples = syll_df.shape[0]

        #select n_examples random syllables from filtered syll_df
        syll_examples = syll_df.sample(n = n_examples, random_state = random_seed)

        for syll_row in syll_examples.itertuples():

            #load file for subject syllable
            file_path = song_folder_path + syll_row.Bird_ID + "/" + syll_row.files
            song = dataloading.SongFile(file_path)
            song.bandpass_filter(500, 15000)

            #set plot title
            title = syll_row.Bird_ID + "    " + syll_row.files

            #plot syllable
            plot_syll(song, syll_row.onsets, syll_row.offsets, padding = padding, figsize = figsize, title = title)

            
