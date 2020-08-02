#!/usr/bin/env python

from __future__ import print_function
from operator import itemgetter

import os
import re
import sys
import time
import math
import glob
import yaml
import h5py
import copy
import cmath
import scipy
import random
import locale
import visdom
import logging
import librosa
import argparse
import itertools
import traceback
import numpy as np
from random import shuffle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import torch.multiprocessing as mp

import scipy
from scipy import signal
from scipy.io import loadmat
from scipy.io import wavfile
from scipy.fftpack import fft, ifft,fftshift

import multiprocessing as mp

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

import dotmap
from dotmap import DotMap

# import pytorch_memlab
# from pytorch_memlab import profile

# import museval
# import speechpy
# import soundfile