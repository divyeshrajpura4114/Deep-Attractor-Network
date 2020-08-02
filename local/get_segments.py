#!/usr/bin/env python3

import os
import sys
import librosa
import argparse

import params

def GetArgs():
    parser = argparse.ArgumentParser(description="Voice Activity Detection (webrtc)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in-data-dir', type=str, dest = "in_data_dir", required = True,
        help='Proivde valid input data directory')
    parser.add_argument('--conf-path', type=str, dest = "conf_path", required = True,
        help='Proivde valid config path')
    sys.stderr.write(' '.join(sys.argv) + "\n")
    args = parser.parse_args()
    args = CheckArgs(args)
    return args

def CheckArgs(args):
    if not os.path.exists(args.in_data_dir):
        raise Exception("This script expects" + args.in_data_dir + "to exist")
    if not os.path.exists(args.conf_path):
        raise Exception("This script expects" + args.conf_path + "to exist")
    return args

def main():
    args = GetArgs()
    hp = params.Hparam(args.conf_path).load_hparam()
    
    max_segment_dur = hp.data.segments.max_segment_dur
    overlap_dur = hp.data.segments.overlap_dur
    
    utt2dur_path = os.path.join(args.in_data_dir, "utt2dur")
    segments_path = os.path.join(args.in_data_dir, "segments")

    if not os.path.exists(utt2dur_path):
        raise Exception("This script expects" + utt2dur_path + "to exist")
    
    utt2dur_dict = {}
    with open(utt2dur_path, "r") as f_utt2dur, open(segments_path, "w") as f_segments:
        for utt2dur_data in f_utt2dur:
            utt_id, utt_dur = utt2dur_data.split(' ')
            utt_dur = float(utt_dur.replace('\n', ''))

            start_segment_dur = 0.0
            end_segment_dur = start_segment_dur + max_segment_dur
            
            while end_segment_dur < utt_dur:
                start_segment_dur_str = str(format(start_segment_dur,'5f')).replace('.','')
                end_segment_dur_str = str(format(end_segment_dur,'5f')).replace('.','')
                f_segments.write("{0:s}_{1:s}_{2:s} {3:s} {4:7.5f} {5:7.5f}\n".format(utt_id, start_segment_dur_str, end_segment_dur_str, utt_id, start_segment_dur, end_segment_dur))
                start_segment_dur = end_segment_dur - overlap_dur
                end_segment_dur = end_segment_dur - overlap_dur + max_segment_dur

if __name__ == '__main__':
    main()