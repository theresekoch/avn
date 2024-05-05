# Changelog

## v.0.5.0 (05/05/2024)

### Feature
- Added Similarity Scoring module

### Note
- The required librosa version was bumped from 0.8.0 to 0.10.2. This version has some bug fixes to the `librosa.yin()` function, which is what AVN uses to calculate pitch in the acoustics module. The calculated pitch values will be slightly different with this new version. See [here] (https://github.com/librosa/librosa/issues/1425) for more information on this change. 

## v0.4.0 (04/01/2024)

### Feature
- Added timing module
- Updated dataloading.SongFile to be compatible with updated librosa version. 

## v0.3.0 (03/10/2023)

### Feature
- Added acoustics module

## v0.2.0 (04/11/2021)

### Feature

- Added syntax module
- Added plotting.Utils.plot_syll_examples(), plotting.plot_spectrogram_with_labels(), plotting.plot_syll(), and plotting.plot_syntax_raster()
    to plotting module. All of these require labeled syllable data. 
- Added dataloading.Utils.select_syll() 