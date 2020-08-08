#!/usr/bin/env python

import common
from common import params, libraries
from common.libraries import *

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

def get_mel_stft(utt_path, utt_start, utt_end, hp):
    s_raw, sr = librosa.core.load(utt_path, sr = None)
    if sr != hp.sr:
        s_raw  = librosa.core.resample(s_raw, sr, hp.sr)
    
    s_raw = s_raw[int(utt_start * hp.sr) : int(utt_end * hp.sr)]
    s_stft = librosa.core.stft(s_raw, n_fft = hp.features.n_fft, 
                                hop_length = hp.features.hop_size, 
                                win_length = hp.features.win_size, 
                                window = hp.features.window)
    
    s_stft = s_stft + float(hp.eps)
    s_abs = np.abs(s_stft)
    s_ph = np.angle(s_stft)
    mel = librosa.filters.mel(sr = hp.sr, n_fft = hp.features.n_fft, 
                                        n_mels = hp.features.n_mels,
                                        fmin = hp.features.fmin,
                                        fmax = hp.features.fmax,
                                        htk = hp.features.htk)
    s_mel_stft = mel.dot(s_abs)
    s_mel_power = np.power(s_mel_stft,2)
    return s_abs.T, s_ph.T, s_mel_stft.T, s_mel_power.T
    
def get_features(segment_index, segment_data, wavscp_dict, featsscp_path, in_data_dir, feat_dir, sources, hp):
    hp = DotMap(hp)
    utt_id, rec_id, utt_start, utt_end = segment_data.split(' ')
    utt_start = float(utt_start)
    utt_end = float(utt_end)
    utt_path = wavscp_dict[rec_id]
    utt_path_org = utt_path.replace('\n','')
    
    in_data_dir = in_data_dir.rstrip('//')
    feats_path = os.path.join(feat_dir, "mfcc_" + os.path.split(in_data_dir)[1] + "_" + str(os.getpid()) + '.hdf5')
    
    s_data_dict = {}
    for source in sources + ["mixture"]:
        utt_path = os.path.join(utt_path_org, source + ".wav")
        s_abs, s_ph, s_mel_stft, s_mel_power = get_mel_stft(utt_path, utt_start, utt_end, hp)
        s_data_dict[source] = {"s_abs": s_abs, "s_ph": s_ph, "s_mel_stft": s_mel_stft, "s_mel_power":s_mel_power}
    
    mixture_mel_power = []
    for source in sources:
        mixture_mel_power.append(s_data_dict[source]["s_mel_power"])
    mixture_mel_power = np.array(mixture_mel_power).sum(axis = 0)
    
    for source in sources + ["mixture"]:
        s_data_dict[source]["ideal_mask"] = np.divide(s_data_dict[source]["s_mel_power"], mixture_mel_power)

    weight_thresh = np.ones(shape = mixture_mel_power.shape)
    
    with h5py.File(feats_path, 'a') as f_feats:
        if utt_id not in f_feats.keys():
            for source in sources + ["mixture"]:
                grp = f_feats.create_group(utt_id + "/" + source) 
                for key, value in s_data_dict[source].items():
                    grp.create_dataset(key, data = value)
            f_feats[utt_id].create_dataset("weight_thresh", data = weight_thresh)

    with open(featsscp_path, "a") as f_featsscp:
        f_featsscp.write("{0:s} {1:s}\n".format(utt_id, feats_path))

def main():
    args = GetArgs()
    hp = params.Hparam(args.conf_path).load_hparam()
    
    feat_dir = "mfcc"

    wavscp_path = os.path.join(args.in_data_dir, "wav.scp")
    segments_path = os.path.join(args.in_data_dir, "segments")
    featsscp_path = os.path.join(args.in_data_dir, "feats.scp")

    if not os.path.exists(segments_path):
        raise Exception("This script expects" + segments_path + " to exist")

    sources = hp.sources.sources
    if hp.sources.num_sources != len(sources):
        raise Exception("The number of sources and provided sources do not match")

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
    pool.starmap_async(get_features, [(segment_index, segment_data, wavscp_dict, featsscp_path, args.in_data_dir, feat_dir, sources, dict(hp)) for segment_index, segment_data in enumerate(segments_list)])
    pool.close()
    pool.join()
    print('That took {} seconds'.format(time.time() - starttime))

if __name__ == '__main__':
    main()