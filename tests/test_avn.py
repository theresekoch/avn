from avn import __version__
import avn.acoustics as acoustics
import avn.dataloading as dataloading
import pandas as pd
import numpy as np

def test_all_features():

    expected= pd.read_csv("C:/Grad_School/Code_and_software/Py_code/avn/tests/all_features_true_avn_windowing.csv")
    song = dataloading.SongFile("C:/Grad_School/Code_and_software/Py_code/avn/tests/G402_43362.23322048_9_19_6_28_42.wav")
    song_acoustics = acoustics.SongInterval(song)
    output = song_acoustics.calc_all_features()

    features = ['Goodness', 'Mean_frequency', 'Entropy', 'Amplitude', 'Amplitude_modulation', 
            'Frequency_modulation', 'Pitch']
    for feature in features:
        assert(np.allclose(expected[feature], output[feature], atol=1e-7))
