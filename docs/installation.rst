Installation 
============
To avoid dependency conflicts, please install `avn` in a fresh conda environment with Python 3.8.
For more information on creating conda environments, refer to the 
`anaconda documentation <https://docs.anaconda.com/anaconda/install/>`_. 
Once you've activated your new conda environment, run::
    pip install avn

The similarity scoring module has some additional dependencies that you will need to install separately
in order to use this module. First, install pytorch in your conda environment according to your system requirements, 
as explained `here <https://pytorch.org/get-started/locally/)>`_. 

Then you will need to install the package that we use for EMD calculations. That can be done by running this 
command from a command prompt with your virtual environment activated:: 
    pip install -e git+https://github.com/theresekoch/pyemd.git#egg=emd

Once you've done that you should be all set! If you run into any issues installing these dependencies, 
please reach out via `email <mailto:therese.koch1@gmail.com>`_ or through `github <https://github.com/theresekoch/avn>`_. 
