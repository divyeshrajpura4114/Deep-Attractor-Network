#!/usr/bin/env python3

import os
import sys
import librosa
import argparse

def GetArgs():
    parser = argparse.ArgumentParser(description="Voice Activity Detection (webrtc)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in-data-dir', type=str, dest = "in_data_dir", required = True,
        help='Proivde valid input data directory')
    sys.stderr.write(' '.join(sys.argv) + "\n")
    args = parser.parse_args()
    args = CheckArgs(args)
    return args

def CheckArgs(args):
    if not os.path.exists(args.in_data_dir):
        raise Exception("This script expects" + args.in_data_dir + "to exist")
    return args

def main():
    args = GetArgs()
    
    wavscp_path = os.path.join(args.in_data_dir, "wav.scp")
    utt2dur_path = os.path.join(args.in_data_dir, "utt2dur")
    
    if not os.path.exists(wavscp_path):
        raise Exception("This script expects" + wavscp_path + "to exist")
    
    with open(wavscp_path, "r") as f_wavscp:
        wav_list = f_wavscp.read().splitlines()

    with open(utt2dur_path, "w") as f_utt2dur:
        for wav_data in wav_list:
            utt_id, utt_path = wav_data.split(' ')

            utt_path = os.path.join(utt_path, "vocals.wav")
            if not os.path.exists(utt_path):
                raise Exception("This script expects" + utt_path + "to exist")

            s_raw, sr = librosa.load(utt_path, sr = None)
            utt_dur = librosa.get_duration(y = s_raw, sr = sr)
            f_utt2dur.write("{0:s} {1:7.5f}\n".format(utt_id, float(utt_dur)))

if __name__ == '__main__':
    main()