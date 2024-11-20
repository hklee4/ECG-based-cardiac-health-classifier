# Code Description

This repository contains the codes and filtered data to generate the main results described in our paper 
"Bio-inspired Monolithically Interlocked Permeable Skin-Adhesive Electrodes for Wireless Vital Diagnostics"




The code in main_STFT.py file loads RAW_DATA for signal pre-processing and transform to 2-D STFT image.

The code in main_peak.py file perform peak detection by loading RAW_DATA files to obtain the SDNN value.

The code in main_classifier.py file perform machine learning using VGG-16 to distinguish the N, AF, ST, and VF.
