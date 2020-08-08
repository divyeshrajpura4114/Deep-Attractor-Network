# Deep Attractor Netowrk for Music Source Separation
____

## Required Python libraries:

```
torch==1.3.1
torchvision==0.1.8.8
librosa==0.8.0
visdom==0.1.8.9
matplotlib==3.3.0
numpy==1.19.1
scipy==1.5.2
sklearn=0.23.2
dotmap==1.3.17
```
These libraries can be installed via following command:

```
pip install -r requirements.txt
```

```diff
- Note: This repository is modified version of the original implementation provided at https://github.com/naplab/DANet by authors  
```

# Citation:

If you use this code for your research, please cite below papers.

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