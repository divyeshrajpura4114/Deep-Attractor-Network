#!/usr/bin/env python

from . import libraries, params
from .libraries import *

class MUSDB18Dataset(Dataset):
    def __init__(self, in_data_dir, hp):
        self.hp = hp
        self.in_data_dir = in_data_dir
        self.featsscp_path = os.path.join(self.in_data_dir, "feats.scp")
        self.featsscp_list = []
        with open(self.featsscp_path, "r") as f_featsscp:
            self.featsscp_list = f_featsscp.read().splitlines()
            shuffle(self.featsscp_list)

    def __len__(self):
        return len(self.featsscp_list)

    def __getitem__(self, index):
        utt_id, utt_feat_path = self.featsscp_list[index].split(' ')
        utt_abs, utt_ph, utt_mel_stft, utt_mel_power, utt_ideal_mask, utt_weight_thresh = self.get_features(utt_id, utt_feat_path)
        return utt_id, (utt_abs, utt_ph, utt_mel_stft, utt_mel_power, utt_ideal_mask, utt_weight_thresh)

    def get_features(self, utt_id, utt_feat_path):
        sources = self.hp.sources.sources
        
        utt_abs = []
        utt_ph = []
        utt_mel_stft = []
        utt_mel_power = []
        utt_ideal_mask = []
        with h5py.File(utt_feat_path, "r") as f_utt_feat:
            for source in sources + ["mixture"]:
                utt_abs.append(f_utt_feat[utt_id][source]["s_abs"][()])
                utt_ph.append(f_utt_feat[utt_id][source]["s_ph"][()])
                utt_mel_stft.append(f_utt_feat[utt_id][source]["s_mel_stft"][()])
                utt_mel_power.append(f_utt_feat[utt_id][source]["s_mel_power"][()])
                utt_ideal_mask.append(f_utt_feat[utt_id][source]["ideal_mask"][()])
            utt_weight_thresh = f_utt_feat[utt_id]['weight_thresh'][()]

        utt_abs = np.array(utt_abs)
        utt_ph = np.array(utt_ph)
        utt_mel_stft = np.array(utt_mel_stft)
        utt_mel_power = np.array(utt_mel_power)
        utt_ideal_mask = np.array(utt_ideal_mask)

        return utt_abs, utt_ph, utt_mel_stft, utt_mel_power, utt_ideal_mask, utt_weight_thresh