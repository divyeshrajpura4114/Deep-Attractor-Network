#!/usr/bin/env python

import params
import libraries
from libraries import *

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

def get_mfcc(segment_index, segment_data, wavscp_dict, featsscp_path, in_data_dir, feat_dir, sources, hp):
    hp = DotMap(hp)
    utt_id, rec_id, utt_start, utt_end = segment_data.split(' ')
    utt_path = wavscp_dict[rec_id]
    utt_path_org = utt_path.replace('\n','')
    
    feats_path = os.path.join(feat_dir, "mfcc_" + in_data_dir.split('/')[-1] + "_" + str(os.getpid()) + '.hdf5')
    
    for source in sources:
        utt_path = os.path.join(utt_path_org, source + ".wav")
        
        s_raw, sr = librosa.core.load(utt_path, sr = None)

        if sr != hp.sr:
            s_raw  = librosa.core.resample(s_raw, sr, hp.sr)

        mfcc = librosa.feature.mfcc(y = s_raw, sr = hp.sr, S = None, n_mfcc = hp.features.n_mfcc, 
                                    n_fft = hp.features.n_fft, hop_length = hp.features.hop_size, 
                                    win_length = hp.features.win_size, window = hp.features.window)
        
        mfcc = mfcc + float(hp.eps)
        mfcc_delta = librosa.feature.delta(mfcc) + float(hp.eps)
        mfcc_delta2 = librosa.feature.delta(mfcc_delta) + float(hp.eps)
        features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2]).T
        
        with h5py.File(feats_path, 'a') as f_feats:
            if utt_id in f_feats.keys():
                f_feats[utt_id].create_dataset(source, data = features)
            else:
                grp = f_feats.create_group(utt_id)
                grp.create_dataset(source, data = features)

    with open(featsscp_path, "a") as f_featsscp:
        f_featsscp.write("{0:s} {1:s}\n".format(utt_id, feats_path))

        print(featsscp_path)

    # if hp.features.cmvn.apply == True:
    #     if hp.features.cmvn.local == False:
    #         features_cmvn = speechpy.processing.cmvn(features, variance_normalization = hp.features.cmvn.var_norm)
    #     else:
    #         features_cmvn = speechpy.processing.cmvnw(features, win_size = hp.features.cmvn.win_size, variance_normalization = hp.features.cmvn.var_norm)

def main():
    args = GetArgs()
    hp = params.Hparam(args.conf_path).load_hparam()

    feat_dir = "mfcc"

    wavscp_path = os.path.join(args.in_data_dir, "wav.scp")
    segments_path = os.path.join(args.in_data_dir, "segments")
    featsscp_path = os.path.join(args.in_data_dir, "feats.scp")

    if not os.path.exists(segments_path):
        raise Exception("This script expects" + semgents_path + "to exist")

    sources = hp.sources

    with open(segments_path, "r") as f_segments:
        segments_list = f_segments.read().splitlines()
    
    with open(featsscp_path, "w") as f_featsscp:
        f_featsscp.truncate(0)

    wavscp_dict = {}
    with open(wavscp_path, "r") as f_wavscp:
        for wav_data in f_wavscp:
            rec_id, rec_path = wav_data.split(' ')
            wavscp_dict[rec_id] = rec_path

    starttime = time.time()
    num_processes = 3
    pool = mp.Pool(processes = num_processes)
    pool.starmap_async(get_mfcc, [(segment_index, segment_data, wavscp_dict, featsscp_path, args.in_data_dir, feat_dir, sources, dict(hp)) for segment_index, segment_data in enumerate(segments_list)])
    pool.close()
    pool.join()
    print('That took {} seconds'.format(time.time() - starttime))

if __name__ == '__main__':
    main()