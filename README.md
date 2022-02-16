# Adaptive artifact subspace reconstruction (AASR)

### Abstract
Demonstration of adaptive artifact subspace reconstruction for artifact removal in EEG. All AASR implement and benchmark test example in MATLAB code.

1. MATLAB (>2019a a.k.a MATLAB runtime > 9.6)
2. Add the path in your project => addpath('AASR')
3. EOG benchmark data from [6] [link to dataset](https://drive.google.com/file/d/1M4v4tV1FNpiC3lnukASm1K0KMiLDw39S/view?usp=sharing)
4. Example code showed in jupyter notebook format



### Usage
1. Check the data filtered by 1-50 Hz bandpass filte
* datafiltering2 is band-pass function function
* datafiltering2(data,channel_sequence,samplingrate); size of data =>[ch, times]
2. Initialization PSW-ASR object
* ASR_PSW(cutoff, samplerate) 
asr_psw = ASR_PSW(20,srate);
3. Select reference segment to calculate artifact
* member function update() = subspace() in first calculation but update() with PSW machanism in the second time and above

4. member function reconstruct() is used for reconstruction of streaming data
* data_processed = asr_psw.reconstruct(test_unclean);


### Reference
1. Delorme A & Makeig S (2004) EEGLAB: an open-source toolbox for analysis of single-trial EEG dynamics, Journal of Neuroscience Methods 134:9-21.
2. asr_calibrate.m and asr_process.m Copyright (C) 2013 The Regents of the University of California Note that this function is not free for commercial use.
3. Clean_rawdata EEGLAB plug-in (https://github.com/sccn/clean_rawdata)
4. Pehlevan, C., Sengupta, A. and Chklovskii, D.B. "Why do similarity matching objectives lead to Hebbian/anti-Hebbian networks?." Neural computation 30, no. 1 (2018): 84-124.
5. online_psp repository (https://github.com/flatironinstitute/online_psp)
6. Klados, Manousos A., and Panagiotis D. Bamidis. "A semi-simulated EEG/EOG dataset for the comparison of EOG artifact rejection techniques." Data in brief
