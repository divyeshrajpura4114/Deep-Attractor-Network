#!/usr/bin/env python

import common
from common import params, libraries
from common.libraries import *

import nnet
from nnet import model, dataset, utils

def GetArgs():
    parser = argparse.ArgumentParser(description="Voice Activity Detection (webrtc)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in-data-dir', type=str, dest = "in_data_dir", required = True,
        help='Proivde valid input data directory')
    parser.add_argument('--model-path', type=str, dest = "model_path", required = True,
        help='Proivde valid model path')
    parser.add_argument('--result-dir', type=str, dest = "result_dir", required = True,
        help='Proivde valid result directory')
    parser.add_argument('--conf-path', type=str, dest = "conf_path", required = True,
        help='Proivde valid config path')
    sys.stderr.write(' '.join(sys.argv) + "\n")
    args = parser.parse_args()
    args = CheckArgs(args)
    return args

def CheckArgs(args):
    if not os.path.exists(args.in_data_dir):
        raise Exception("This script expects " + args.in_data_dir + " to exist")
    if not os.path.exists(args.conf_path):
        raise Exception("This script expects " + args.conf_path + " to exist")
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(args.model_path):
        raise Exception("This script expects " + args.model_path + " to exist")
    return args

def do_testing():
    danet_model.eval()
    for batch_id, (batch_utt_id, batch_data) in enumerate(test_loader):
        batch_mixture_abs, batch_mixture_ph, batch_mixture_mel_stft, batch_ideal_mask, batch_weight_thresh = batch_data
        print(batch_mixture_abs.shape)
        print(batch_utt_id)
        print(batch_ideal_mask)

        with torch.no_grad():
            hidden = danet_model.init_hidden(hp.test.batch_size)
            batch_est_mask = danet_model(hidden = hidden, batch_features = batch_mixture_mel_stft, batch_weight_thresh = batch_weight_thresh, batch_ideal_mask = batch_ideal_mask)

        for utt_index, utt_est_masks in enumerate(batch_est_mask):
            utt_id = batch_utt_id[utt_index]

            utt_result_dir = os.path.join(args.result_dir, utt_id)
            if not os.path.exists(utt_result_dir):
                os.makedirs(utt_result_dir)
    
            # estimate attractors via K-means
            utt_est_masks = utt_est_masks.data.cpu().numpy()  # T*F, K
            kmeans_model = KMeans(n_clusters = len(sources), random_state = 0).fit(utt_est_masks.astype('float64')) 
            utt_attractors = kmeans_model.cluster_centers_ # nspk, K
            
            # estimate masks
            utt_est_masks = torch.from_numpy(utt_est_masks).float().to(hp.device)  # T*F, K
            utt_attractors = torch.from_numpy(utt_attractors.T).float().to(hp.device)  # K, nspk

            utt_est_masks = F.softmax(torch.mm(utt_est_masks, utt_attractors), dim=1)  # T*F, nspk

            utt_est_masks = utt_est_masks.data.cpu().numpy()  # T*F, K
            
            for source_index, source in enumerate(sources):
                utt_est_mask_source = utt_est_masks[:,source_index].reshape(-1, hp.model.feature_size)
                
                utt_est_mask_source = librosa.feature.inverse.mel_to_stft(M = utt_est_mask_source.T, sr = hp.sr, n_fft = hp.features.n_fft, fmin = hp.features.fmin, fmax = hp.features.fmax, htk = hp.features.htk).T
                
                # masking the mixture magnitude spectrogram
                utt_est_spec_source = (batch_mixture_abs[source_index].cpu().numpy() * utt_est_mask_source) * np.exp(1j*batch_mixture_ph[source_index].cpu().numpy())
                utt_est_s_raw_source = librosa.core.istft(utt_est_spec_source.T, hop_length = hp.features.hop_size, win_length = hp.features.win_size)
                librosa.output.write_wav(os.path.join(utt_result_dir, source + ".wav" ), utt_est_s_raw_source, hp.sr)

if __name__ == '__main__':
    args = GetArgs()
    hp = params.Hparam(args.conf_path).load_hparam()
    sources = hp.sources.sources

    featsscp_path = os.path.join(args.in_data_dir, "feats.scp")
    if not os.path.exists(featsscp_path):
        raise Exception("This script expects" + featsscp_path + "to exist")

    danet_model = model.DANet(hp, "test").to(hp.device)
    
    danet_model.load_state_dict(torch.load(args.model_path, map_location = hp.device)['model_state_dict'])
    
    test_data = dataset.MUSDB18Dataset(args.in_data_dir, hp)
    test_loader = DataLoader(test_data, batch_size = hp.test.batch_size, shuffle = True, drop_last = True, collate_fn = utils.load_train_batch)

    do_testing()