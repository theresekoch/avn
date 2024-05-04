"""
Created on Fri May 3 2024

@author: Therese
"""

import numpy as np
import pandas as pd
import librosa
import avn.dataloading
import torchvision.datasets as datasets
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from emd import emd


def _make_embedding_spect(syll_wav, sr, hop_length, win_length, n_fft, amin, ref_db, min_level_db, 
                          low_bandpass = None, high_bandpass = None):
    #make spectrogram of file
    spectrogram = librosa.stft(syll_wav, hop_length = hop_length, win_length = win_length, n_fft = n_fft)
    #convert to db scale
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), amin = amin, ref = ref_db)
    
    #normalize
    S_norm = np.clip((spectrogram_db - min_level_db) / -min_level_db, 0, 1)

    #cut frequencies outside range if range is provided
    if ~ (low_bandpass == None):
        frequency_list = librosa.fft_frequencies(sr = sr, n_fft = n_fft)
        low_bandpass_idx = np.argwhere(frequency_list < low_bandpass).max()
        high_bandpass_idx = np.argwhere(frequency_list > high_bandpass).min()
        
        S_norm = S_norm[low_bandpass_idx:high_bandpass_idx, :]
    
    return S_norm

def prep_spects(Bird_ID, segmentations, song_folder_path, out_dir, n_files = None, 
                low_bandpass = 2000, high_bandpass = 6000, 
                hop_length = 128, win_length = 512, n_fft = 512, amin = 1e-5,
                ref_db = 20, min_level_db = -28, pad_length = 70):
    """Make and save spectrograms of segmented syllables for embedding

   Calculate and save a spectrogram of each syllable in `segmentations` for embedding with the 
   similarity scoring embedding model. All optional parameters should be left unchanged for compatibility 
   with embedding model. 

    :param Bird_ID: ID of the current bird to be processed. Will be used to name spectrogram files. 
    :type Bird_ID: str
    :param segmentations: pandas dataframe of syllable segmentations, with one row per syllable, and columns called :
    - 'files' : the name of the .wav file in which a syllable is found, 
    - 'onsets' : the onset of the syllable within the .wav file in seconds, and 
    - 'offsets': the offset of the syllable within the .wav file in seconds. 
    
    We recommend using [WhisperSeg](https://github.com/nianlonggu/WhisperSeg) to automatically segment song syllables and generate this type of table.

    :type segmentations: pd.DataFrame
    :param song_folder_path: path to the folder containing all wav files from `segmentations`.
    :type song_folder_path: str
    :param out_dir: path to the folder where you want to save all the spectrograms. Each bird's spectrograms must be saved to a unique directory 
        for use with embedding model. 
    :type out_dir: str
    :param n_files: maximum number of files from which to make spectrograms. By default, spectrograms of all syllables will be made. 
        If `n_files` is specified and `segmentations` contains syllables from more than `n_files` unique files, then a random subset 
        of `n_files` will be selected and only syllables from those files will have spectrograms made. Setting this value can help save 
        time and memory when you have more than enough song files for reliable EMD scoring (>>500 .wav files). 
    :type n_files: int>0, optional
    :param low_bandpass: lower frequency cutoff for spectrograms, defaults to 2000
    :type low_bandpass: int>0, optional
    :param high_bandpass: upper frequency cutoff for spectrograms, defaults to 6000
    :type high_bandpass: int>low_bandpass, optional
    :param hop_length: number of audio samples between adjacent stft columns, defaults to 128
    :type hop_length: int>0, optional
    :param win_length: number of audio samples in each window for stft calculation, defaults to 512
    :type win_length: int>0, optional
    :param n_fft: length of window after 0 padding for stft calculation, defaults to 512
    :type n_fft: int>0, optional
    :param amin: minimum threshold for spectrogram when converting to db, defaults to 1e-5. See librosa.amplitude_to_db. 
    :type amin: float>0, optional
    :param ref_db: reference amplitude for scaling spectrogram to db, defaults to 20. See librosa.amplitude_to_db. 
    :type ref_db: float >-=, optional
    :param min_level_db: minimum decibel value for normalization of db spectrogram, defaults to -28
    :type min_level_db: int, optional
    :param pad_length: dimension of output spectrogram in frames, defaults to 70. spectrograms longer than 70 frames will 
        be clipped to 70 frames, and spectrograms shorter than 70 frames will be padded to 70 frames.
    :type pad_length: int>0, optional
    :return: None
    :rtype: None
    """
    #initialize df for audio
    syllable_dfs = pd.DataFrame()

    #if n_files is specified, sample n_files. Otherwise use all files
    if n_files is not None: 
        if len(segmentations.files.unique()) > n_files:
            song_files = np.random.choice(segmentations.files.unique(), size = n_files, replace = False)
        else:
            song_files = segmentations.files.unique()
    else: 
        song_files = segmentations.files.unique()
    
    #loop over each song file and create spectrograms. 
    for song_file in song_files:
        file_path = song_folder_path + song_file
        song = avn.dataloading.SongFile(file_path)
        song.bandpass_filter(low_bandpass, high_bandpass)

        syllable_df = segmentations[segmentations['files'] == song_file].copy()

        #this section is based on avgn.signalprocessing.create_spectrogram_dataset.get_row_audio()
        syllable_df["audio"] = [song.data[int(st * song.sample_rate) : int(et * song.sample_rate)]
                               for st, et in zip(syllable_df.onsets.values, syllable_df.offsets.values)]
        syllable_dfs = pd.concat([syllable_dfs, syllable_df])
        
    #Normalize the audio
    syllable_dfs['audio'] = [librosa.util.normalize(i) for i in syllable_dfs.audio.values]
    
    #initialize list for syllable spectrograms
    syllables_spec = []

    #compute spectrogram for each syllable
    for syllable in syllable_dfs.audio.values:
        
        syllable_spec = _make_embedding_spect(syllable, 
                                              sr = song.sample_rate,
                                                hop_length = hop_length, 
                                                win_length = win_length, 
                                                n_fft = n_fft, 
                                                ref_db = ref_db, 
                                                amin = amin, 
                                                min_level_db = min_level_db, 
                                                low_bandpass=low_bandpass, 
                                                high_bandpass=high_bandpass)

        syllables_spec.append(syllable_spec)
        
    #normalize spectrograms
    def norm(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    syllables_spec_norm = [norm(i) for i in syllables_spec]

    #create output directories if they don't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    
    #Pad spectrograms for uniform dimensions and save. 
    
    for i, spec in enumerate(syllables_spec_norm):
        if np.shape(spec)[1]> pad_length: 
            spec_padded = spec[:, :pad_length]
        else: 
            to_add = pad_length - np.shape(spec)[1]
            pad_left = np.floor(float(to_add) / 2).astype("int")
            pad_right = np.ceil(float(to_add) / 2).astype("int")
            spec_padded = np.pad(spec, [(0, 0), (pad_left, pad_right)], 'constant', constant_values = 0)
        
        file_name = Bird_ID + "_" + str(i).zfill(4)

        out_file_path = out_dir + "/" + file_name

        np.save(out_file_path, spec_padded)

    print(str(len(syllables_spec_norm))  + " syllable spectrograms saved in " + out_dir)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class InceptionBlock(nn.Module):
    def __init__( self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, red_7x7, out_7x7):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
                                     ConvBlock(in_channels, red_3x3, kernel_size=1, padding=0), 
                                     ConvBlock(red_3x3, out_3x3, kernel_size = 3, padding = 1), )

        self.branch3 = nn.Sequential(
                                    ConvBlock(in_channels, red_5x5, kernel_size = 1), 
                                    ConvBlock(red_5x5, out_5x5, kernel_size = 5, padding = 2), )

        self.branch4 = nn.Sequential(
                                     ConvBlock(in_channels, red_7x7, kernel_size = 1, padding = 0), 
                                     ConvBlock(red_7x7, out_7x7, kernel_size = 7, padding = 3), )

        
    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        return torch.cat([branch(x) for branch in  branches],  1)
    
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3), nn.ReLU())
        self.MAM1 = InceptionBlock(32, 32, 16, 32, 16, 32, 16, 32)
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size= 3, stride = (2, 1)), nn.ReLU())
        self.MAM2 = InceptionBlock(64, 32, 16, 32, 16, 32, 16, 32)
        self.conv3 = nn.Sequential(nn.Conv2d(128, 64, 3, stride = (2, 1)), nn.ReLU())
        self.MAM3 = InceptionBlock(64, 32, 16, 32, 16, 32, 16, 32)
        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride = (2, 1)), nn.ReLU())
        self.MAM4 = InceptionBlock(64, 32, 16, 32, 16, 32, 16, 32)
        self.conv5 = nn.Sequential(nn.Conv2d(128, 64, 3, stride = (2, 1)), nn.ReLU())

        self.global_pool = nn.AvgPool2d(kernel_size = (1, 4), stride = 4)

        self.fc = nn.Sequential(nn.Linear(960, 256),
                                nn.ReLU(),
                                 
                                nn.Linear(256, 128),
                                nn.ReLU(),
                                 
                                nn.Linear(128, 8)
                                )


    def forward(self, x):
        x = torch.unsqueeze(x, 1).float()
        x = self.conv1(x)
        x = self.MAM1(x)
        x = self.conv2(x)
        x = self.MAM2(x)
        x = self.conv3(x)
        x = self.MAM3(x)
        x = self.conv4(x)
        x = self.MAM4(x)
        x = self.conv5(x)
    
        x = self.global_pool(x)
        x = x.view(x.size()[0], -1)

        x = self.fc(x)

        x = F.normalize(x, p = 2, dim = 1)
        

        return x
        
    def get_embedding(self, x):
        return self.forward(x)
    

def load_model(device = 'auto'):
    """load embedding model

    Load embedding model. 

    :param device: device on which to load and run embedding model, defaults to 'auto'. 
        If 'auto', device will be set to 'cuda' if available and 'cpu' otherwise. 
    :type device: ['auto', 'cpu', or 'cuda'], optional
    :return: model
    :rtype: avn.similarity.EmbeddingNet
    """
    #check device
    if device is 'auto':
        cuda = torch.cuda.is_available()
        if cuda:
            device = 'cuda'
        else: 
            device = 'cpu'
        print('Device set to: ' + device)

    #specify model architecture 
    model = EmbeddingNet()
    #load model weights
    model.load_state_dict(torch.load('..\\8D_trained_embedding_model.pth', map_location = torch.device(device)))
    model.to(device)
    model.device = device
    model.eval()

    return model



class CustomDatasetFolderNoClass(datasets.DatasetFolder):
    """creates dataset compatible with embedding net evaluation functions for unlabeled syllable spectrograms. 
    
    """
    def __init__(self, 
                root: str,
                loader: Callable[[str], Any],
                extensions: Optional[Tuple[str, ...]] = None,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                is_valid_file: Optional[Callable[[str], bool]] = None, 
                Bird_ID: Optional[str] = None, 
                train: Optional[bool] = None, 
                exclude: Optional[list] = None):

        self.train = train
        self.Bird_ID = Bird_ID
        self.exclude = exclude

        if self.train is not None:
            if self.Bird_ID is None:
                raise ValueError("If a train value is provided, a Bird_ID must also be provided")

        if self.train is None: 
            if self.Bird_ID is not None: 
                raise ValueError("For train/test splitting a value for `train` must be provided in addition to `Bird_ID`.")

        super().__init__(root, 
                        loader = loader, 
                        extensions = extensions, 
                        transform = transform, 
                        target_transform = target_transform, 
                        is_valid_file = is_valid_file
                        )

        
        
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = ['']

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def _embeddings_from_dataloader(dataloader, model, n_dim=8):

    device = model.device
    if device == 'cuda':
        cuda = True
    else: 
        cuda = False

    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), n_dim))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            k += len(images)
    return embeddings

def calc_embeddings(Bird_ID, spectrograms_dir, model):
    """calculates syllable embeddings

    Calculates syllable embeddings for all syllables in `spectrograms_dir` using `model`. 
    Compatible spectrograms must be generated from a segmentation table using `similarity.prep_spects()`
    and saved to `spectrograms_dir` before embeddings can be calculated. The model also needs to be loaded 
    using `similarity.load_model()`. 

    :param Bird_ID: ID of the current bird to be processed.
    :type Bird_ID: str
    :param spectrograms_dir: path to folder with prepared spectrograms
    :type spectrograms_dir: str
    :param model: trained model for embedding calculation. Can be loaded with `similarity.load_model()`
    :type model: avn.similarity.EmbeddingNet()
    :return: matrix of syllable embeddings. Will have shape (n_syllables, 8). 
    :rtype: np.array
    """
    #prepare dataloader
    single_bird_dataset = CustomDatasetFolderNoClass(spectrograms_dir, extensions = (".npy"), loader = np.load, 
                                    train = False, Bird_ID = Bird_ID)
    single_bird_dataloader = torch.utils.data.DataLoader(single_bird_dataset, batch_size=64, shuffle=False)
    #get embeddings
    single_bird_embeddings = _embeddings_from_dataloader(single_bird_dataloader, model, n_dim = 8)

    return single_bird_embeddings

def calc_emd(bird_1_embedding, bird_2_embedding):
    """calculate EMD between embeddings

    Calculates the earth mover's distance between two sets of syllable embeddings, each generated with 
    `similarity.calc_embeddings()`. Higher scores indicate less similar songs between the two sets. 

    :param bird_1_embedding: array of shape (n_syllables, 8) with syllable embeddings from one bird to be compared
    :type bird_1_embedding: np.array
    :param bird_2_embedding: array of shape (n_syllables, 8) with syllable embeddings from the other bird to be compared
    :type bird_2_embedding: np.array
    :return: emd score
    :rtype: float
    """
    return emd(bird_1_embedding, bird_2_embedding)