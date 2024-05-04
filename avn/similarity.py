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
    """prepare spectrogram of single wav file for similarity embedding

    _extended_summary_

    :param syll_wav: _description_
    :type syll_wav: _type_
    :param hop_length: _description_
    :type hop_length: _type_
    :param win_length: _description_
    :type win_length: _type_
    :param n_fft: _description_
    :type n_fft: _type_
    :param amin: _description_
    :type amin: _type_
    :param ref_db: _description_
    :type ref_db: _type_
    :param min_level_db: _description_
    :type min_level_db: _type_
    :param low_bandpass_idx: _description_, defaults to None
    :type low_bandpass_idx: _type_, optional
    :param high_bandpass_idx: _description_, defaults to None
    :type high_bandpass_idx: _type_, optional
    :return: _description_
    :rtype: _type_
    """
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

    if device is 'auto':
        cuda = torch.cuda.is_available()
        if cuda:
            device = 'cuda'
        else: 
            device = 'cpu'
        print('Device set to: ' + device)
            
    model = EmbeddingNet()
    
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
    
    single_bird_dataset = CustomDatasetFolderNoClass(spectrograms_dir, extensions = (".npy"), loader = np.load, 
                                    train = False, Bird_ID = Bird_ID)

    single_bird_dataloader = torch.utils.data.DataLoader(single_bird_dataset, batch_size=64, shuffle=False)

    single_bird_embeddings = _embeddings_from_dataloader(single_bird_dataloader, model, n_dim = 8)

    return single_bird_embeddings


def sample_embedding_rows(embedding_array, num_samples):
    if embedding_array.shape[0] > num_samples:
        return embedding_array[np.random.choice(embedding_array.shape[0], num_samples, replace = False)]

    else: 
        return embedding_array[np.random.choice(embedding_array.shape[0], num_samples, replace = True)]


def calc_emd(bird_1_embedding, bird_2_embedding):
    return emd(bird_1_embedding, bird_2_embedding)