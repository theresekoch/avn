.. avn documentation master file, created by
   sphinx-quickstart on Tue May 11 10:56:05 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to avn's documentation!
===============================
`avn` (Avian Vocalization Network, pronounced 'avian') is a python package for zebra finch song analysis. It currently provides
functions necessary for threshold-based syllable segmentation, syntax analysis for songs with user-provided syllable labels, syllable timing and song rhythm feature calculation, and calculation of a suite of acoustic features based on `Sound Analysis Pro <http://soundanalysispro.com/>`_. For more information, 
please contact `Therese Koch <mailto:therese.koch@utsouthwestern.edu>`_ , consult 
the `github repository <https://github.com/theresekoch/avn>`_ , or check out our `pre-print <https://doi.org/10.1101/2024.05.10.593561>`_ !


.. toctree::
   :caption: Getting Started
   :maxdepth: 1

   installation
   AVN_GUI

.. toctree::
   :caption: Tutorials
   :maxdepth: 2

   tutorial
   syntax_analysis_demo
   acoustic_feature_demo
   timing_analysis_demo
   Similarity_scoring_demo


.. toctree::
   :caption: Documentation
   :maxdepth: 3

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
