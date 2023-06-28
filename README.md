# Requirements (cpu installation; windows or linux)
conda install pytorch torchvision torchaudio cpuonly -c pytorch\
conda install botorch -c pytorch -c gpytorch -c conda-forge

# How to use
For running the code, simply run "python qei.py". The experiment can be controlled by modifying the main method (currently set to run the first experiment for CA-qEI). For documentation of the parameters, please read the comments in the helper classes.

# Future changes
Visualizations will need a small readjustment to run smoothly. The config file will also be added in the future.

