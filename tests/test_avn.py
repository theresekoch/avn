from avn import __version__
import avn.acoustics as acoustics
import avn.dataloading as dataloading
import avn.similarity as similarity
import pandas as pd
import numpy as np

def test_all_features():

    expected= pd.read_csv("yin_updated_all_features_true_avn_windowing.csv")
    song = dataloading.SongFile("G402_43362.23322048_9_19_6_28_42.wav")
    song_acoustics = acoustics.SongInterval(song)
    output = song_acoustics.calc_all_features()

    features = ['Goodness', 'Mean_frequency', 'Entropy', 'Amplitude', 'Amplitude_modulation', 
            'Frequency_modulation', 'Pitch']
    for feature in features:
        assert(np.allclose(expected[feature], output[feature], atol=1e-7))


def test_embedding_model():
    expected = pd.read_csv('G402_embeddings.csv').drop(columns = 'Unnamed: 0')
    model = similarity.load_model()
    embeddings = similarity.calc_embeddings(Bird_ID = 'G402', 
                                            spectrograms_dir= '../sample_data/G402_embedding_spectrograms/', 
                                            model = model)
    assert(np.allclose(expected.values, embeddings, atol=1e-3))

def test_emd():
    embeddings = pd.read_csv('G402_embeddings.csv').drop(columns = 'Unnamed: 0')
    emd = similarity.calc_emd(embeddings[:200], embeddings[200:])

    assert(emd == 0.30965134896858937)