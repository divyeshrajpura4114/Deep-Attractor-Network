# Deep Attractor Netowrk for Music Source Separation
____

```diff
- Note: This reposatory has been updated with our latest paper "MSpeC-Net : Multi-Domain Speech Conversion Network".
```
- Demo link: [Click Here](https://maitreyapatel.github.io/mspec-net-demo/)

### Improving the intelligibility of Non-Audible-Murmur and Whisper speech via following different types of conversations:
1. NAM-to-WHiSPer (NAM2WHSP)
2. WHiSPer-to-SPeeCH (WHSP2SPCH)
3. NAM-to-SPeeCH (NAM2SPCH)

### Why it is important to improve intelligibility of such speeches via speech conversion?
The patients that are suffering from the vocal fold paralysis, neurological disorders etc. may not be able to produce normal speech  due  to  the  partial  or  complete  absence  of  vocal  fold vibrations (i.e., voicing). Losing the natural way of producing the  speech  will  affect  one’s  life  extremely,  since  speech  is the most natural and powerful form of communication among humans.  Furthermore, interesting  applications  of  the  whispered speech communications are, private conversation in public using cell phone, conversation in quiet environments like a library, a hospital, a meeting room, etc.  Therefore, current trend lies on developing silent speech interfaces. And to achieve this we suggest to focus on Non-Audible-Murmur (NAM) and Whisper Speech.

Hence,  the  aim  of  the  present  work  is  to  improve intelligibility of NAM and whispered speech via normal speech conversion using Machine Learning(ML)-based  approaches  in  order  to  improve  the  quality  of communication.

For more details please refer below mentioned research papers.

### Published Papers:
| Index | Type of Conversion            | Paper Title                                                                         | Paper-Link | Demo-Link |
|-------|-------------------------------|-------------------------------------------------------------------------------------|------|-----------|
| (1)   | NAM2WHSP, WHSP2SPCH, NAM2SPCH | MSpeC-Net: Multi-Domain Speech Conversion Network                                   |  [[Link]](https://ieeexplore.ieee.org/document/9052966/)   | [[Link]](https://maitreyapatel.github.io/mspec-net-demo/)          |
| (2)   | WHSP2SPCH                     | Novel Inception-GAN for Whisper-to-Normal Speech Conversion                         |   [[Link]](https://www.isca-speech.org/archive/SSW_2019/abstracts/SSW10_P_1-9.html)   |      NA     |
| (3)   | WHSP2SPCH                     | Effectiveness of Cross-Domain Architectures for Whisper-to-Normal Speech Conversion |   [[Link]](https://ieeexplore.ieee.org/abstract/document/8902961)   |    NA      |
| (4)   | WHSP2SPCH                     | Novel mmse discogan for cross-domain whisper-to-speech conversion                   |   [[Link]](https://drive.google.com/file/d/1UVbXRzpaM1_ayaTfvq92RVijn_M8FLnn/view)   |    NA      |

### Proposed Methods:
- MMSE-GAN
- DiscGAN
- CycleGAN
- CNN-GAN
- Inception-GAN
- MSpeC-Net


## Prerequisites:

- Linux, MacOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- SoX (Command line utility that can convert various formats of computer audio files in to other formats.)
- MatLab


### Before going ahead please create the two directories (dataset and results) with follwing sctructre:

```
dataset
|
└───features
    └───US_102
    |   └───batches
    |   |   |    f0
    |   |   |    mcc
    |   |   |    VUV
    |   |
    |   └───Normal
    |   |   |    f0
    |   |   |    mcc
    |   |   |    fv
    |   |
    |   └───Whisper
    |       |    f0
    |       |    mcc
    |       |    fv
    └───MSpeC-Net
        └───batches
        |   |    f0
        |   |    mcc
        |   |    VUV
        |
        └───Normal
        |   |    f0
        |   |    mcc
        |   |    fv
        |
        └───Whisper
        |   |    f0
        |   |    mcc
        |   |    fv
        |
        └───NAM
            |    f0
            |    mcc
            |    fv
```
```
results
|
└───checkpoints
|   |   mcc
|   |   f0
|   |   vuv
|
└───mask
|   |   mcc
|   |   f0
|   |   vuv
|
└───converted_f0
|
└───converted_wav
|
└───mcc
|
└───tmp

```

## Required Python libraries:

```
torchvision==0.2.1
visdom==0.1.8.8
matplotlib==3.1.2
numpy==1.17.4
scipy==1.3.2
torch==1.3.1
```
These libraries can be installed via following command:

```
pip install -r requirements.txt
```


## Dataset:
For whisper-to-normal speech conversion, we rely on wTIMIT dataset. Please download this dataset from [here](http://www.isle.illinois.edu/sst/data/wTIMIT/). And follow below steps:

1. Download the dataset put the .wav files in the `dataset/data/{speaker_name}/{Normal/Whisper}`.
2. In this GitHub repository we use Ahocoder to extract MCEP/MCC features. However, Ahocoder accepts input .wav files with 16,000 Hz frequency only. Thereofore, run following two terminal commands to each .wav files.
    - `sox {path + file1.wav} -t -wav {path + file1.wav}` (Only if downloaded files has .WAV extention.)
    - `sox {path + file1.wav} -r 16000 -c 1 -b 16 {path + file1.wav}`

For MSpeC-Net related experiments we use CSTR-NAM-TIMIT dataset, which is available [here](https://homepages.inf.ed.ac.uk/jyamagis/page3/page57/page57.html). And follow below steps:
1. Download the dataset and put the .wav files in the `dataset/data/MSpeC-Net/{Normal/Whisper/NAM}/wav`
2. Follow the same 2nd step described above. 
   
# Now, do the training for WHSP2SPCH conversion:

Either run the follwing single shell script or follow the steps described below:

```
./run.sh
```

**OR Follow these steps**


## Feature extraction:

Now, run follwing commads in the terminal:

1. `cd scripts`
2. `./feature_extraction.sh`

## Training Batches creation:

Run below mentioned MatLab scripts in **order**.
- concatenation_VCC2018.m
- batch.m
- concatenation_f0.m
- batch_f0.m
- concatenation_vuv.m
- batch_vuv.m

## Training of Deep Learning mapping function:

Yay! Finally it's time to do the training. Now, go to `py_src` directory and run below mentioned commads in terminal.

1. `python -m visdom.server`
2. `python MMSE_GAN.py -tr -te`
3. `python MMSE_GAN_F0.py -tr -te`
4. `python DNN_vuv.py -tr -te`

**Note:**  These python scripts are highly customizable according to the user choises. Such as traininng/testing, parallel/non-parallel training, paths to the features/checkpoint, learning rate, number of epoch, validation interval, check point interval etc.

## Obective measures:

As decribed in the research papers, we use MCD and F0_RMSE as objective evalution metrc. Hence, run follwing two MatLab scripts:

- Testing_MCD.m
- f0obective_cal.m

## Speech Synthesis:

Finally, we are at the last stage. We will use converted mcc and predicted F0 feature as an input to Ahocoder and it will generate the speech from these features. Hence, run follwing command in terminal (scripts directory):

1. `./synthesis.sh`


# Let's take an overview on how to use this library for MSpeC-Net training/testing:

Currently, this repository provides the necessary codes for MSpeC-Net training and testing.
One can train MSpeC-Net model by running following python script:

1. `python MSpeC-Net.py -tr -te`

However, for more details on how to use different matlab scripts and feature extraction shell scripts please refer the steps shown in WHSP2SPCH conversions. 
Righ now, we are working on it to make the final end-to-end system. Soon we will publish it. 

If you are interested in contributing, please get in touch with author of the repository.



# Citation:

If you use this code for your research, please cite our papers.

```
@ARTICLE{YiLuo2018IEEETransASLP,
    author={Y. {Luo} and Z. {Chen} and N. {Mesgarani}},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
    title={Speaker-Independent Speech Separation With Deep Attractor Network},
    year={Apr, 2018},
    volume={26},
    number={4},
    pages={787-796},
    doi={10.1109/TASLP.2018.2795749},
    ISSN={2329-9304},
}
@INPROCEEDINGS{ZhuoChen2017ICASSP, 
    author={Z. {Chen} and Y. {Luo} and N. {Mesgarani}}, 
    booktitle={2017 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)}, 
    title={Deep attractor network for single-microphone speaker separation}, 
    address={New Orleans, Louisiana, USA},
    year={Mar 05-09, 2017},
    volume={}, 
    number={}, 
    pages={246-250}, 
    doi={10.1109/ICASSP.2017.7952155}, 
    ISSN={2379-190X}, 
}

@inproceedings{RajathKumar2018Interspeech,
    author={Rajath Kumar and Yi Luo and Nima Mesgarani},
    title={Music Source Activity Detection and Separation Using Deep Attractor Network},
    Address = {Hyderabad, India},
    year={Sep 02-06, 2018},
    booktitle={Proc. INTERSPEECH 2018},
    pages={347--351},
    doi={10.21437/Interspeech.2018-2326},
}

@INPROCEEDINGS{DivyeshRajpura2020SPCOM,
    author={Divyesh G. Rajpura and Jui Shah and Maitreya Patel and Harshit Malaviya and Kirtana Phatnani and Hemant A. Patil},
    booktitle={accepted in 2020 International Conference on Signal Processing and Communications (SPCOM)}, 
    title={Effectiveness of Transfer Learning on Singing Voice Conversion in the Presence of Background Music}, 
    address={Bengaluru, India},
    year={Jul 20-23, 2020},
    volume={},
    number={},
    pages={1-5},
}
```