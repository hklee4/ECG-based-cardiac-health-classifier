# Code Description

This repository contains the code and filtered data used to generate the main results described in our paper submitted to Nature Materials. 

The code in '1.peak_detection_algorithm.py' file perform peak detection by loading RAW_DATA files to obtain the SDNN value.

The code in '2.Signal_STFT.py' file loads RAW_DATA for signal pre-processing and transform to 2-D STFT image.

The code in '3.image_classifier.py'  file perform machine learning using VGG-16 by using training imagesets.

The code in '4.Load_model_and_Inference.py' file loads test imagesets and hdf5 file to distinguish the N, AF, ST, and VF..


# Signal data 

'Normal_MIT-BIH.csv' is from ‘MIT-BIH Normal Sinus Rhythm Database’

'AF_MIT-BIH.csv' is from ‘MIT-BIH Atrial Fibrillation Database’

'ST_MIT-BIH.csv' is from ‘European ST-T Database’

'VF_MIT-BIH.csv' is from ‘MIT-BIH Malignant Ventricular Ectopy Database’

'Normal_Clinical.csv' is from our wearable device
