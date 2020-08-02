#!/usr/bin/env python

from . import libraries, params
from .libraries import *

hp = params.Hparam().load_hparam()

def load_train_batch(batch_data):
    device = torch.device(hp.device)
    
    batch_utt_id = []
    batch_mixture_abs = []
    batch_mixture_ph = []
    batch_mixture_mel_stft = []
    batch_ideal_mask = []
    batch_weight_thresh = []
    
    for utt_index in range(len(batch_data)):        
        utt_id, utt_data = batch_data[utt_index]
        
        batch_utt_id.append(utt_id)
        batch_mixture_abs.append(utt_data[0][-1])
        batch_mixture_ph.append(utt_data[1][-1])
        batch_mixture_mel_stft.append(utt_data[2][-1])
        batch_ideal_mask.append(utt_data[4][:-1])
        batch_weight_thresh.append(utt_data[5])
    
    batch_ideal_mask = np.array(batch_ideal_mask)
    b, s, f, t = batch_ideal_mask.shape
    batch_ideal_mask = batch_ideal_mask.reshape(b, f*t, s)
    
    batch_weight_thresh = np.array(batch_weight_thresh)
    b, f, t = batch_weight_thresh.shape
    batch_weight_thresh = batch_weight_thresh.reshape(b, f*t, 1)
    
    batch_mixture_abs = torch.Tensor(batch_mixture_abs).float().contiguous().to(device)
    batch_mixture_ph = torch.Tensor(batch_mixture_ph).float().contiguous().to(device)
    batch_mixture_mel_stft = torch.Tensor(batch_mixture_mel_stft).float().contiguous().to(device)
    batch_ideal_mask = torch.Tensor(batch_ideal_mask).float().contiguous().to(device)
    batch_weight_thresh = torch.Tensor(batch_weight_thresh).float().contiguous().to(device)
    
    return batch_utt_id, (batch_mixture_abs, batch_mixture_ph, batch_mixture_mel_stft, batch_ideal_mask, batch_weight_thresh)
    
# def load_train_batch(batch_data):
#     device = torch.device(hp.device)
    
#     batch_mix_features = Variable(batch_data[0]).float().contiguous().to(device)
#     batch_mix_abs = Variable(batch_data[1]).float().contiguous().to(device)
#     batch_mix_ph = Variable(batch_data[2]).float().contiguous().to(device)
#     batch_weight_thresh = Variable(batch_data[3]).float().contiguous().to(device)
#     batch_ideal_mask = Variable(batch_data[4]).float().contiguous().to(device)

#     if train_bool == True:
#         return batch_mix_features, batch_mix_abs, batch_mix_ph, batch_weight_thresh, batch_ideal_mask
#     else:
            
#         batch_vocal_abs = Variable(batch_data[5]).float().contiguous().to(device)
#         batch_vocal_ph = Variable(batch_data[6]).float().contiguous().to(device)
#         batch_inst_abs = Variable(batch_data[7]).float().contiguous().to(device)
#         batch_inst_ph = Variable(batch_data[8]).float().contiguous().to(device)
#         return batch_mix_features, batch_mix_abs, batch_mix_ph, batch_weight_thresh, batch_ideal_mask,  batch_vocal_abs, batch_vocal_ph, batch_inst_abs, batch_inst_ph


def pad_collate(batch):
    new_batch = []
    for index1, val1 in enumerate(batch):
        for index2, val2 in enumerate(val1):
            new_batch.append(batch[index1][index2].T)
    lens = list(map(len, new_batch))
    padded = pad_sequence(new_batch, batch_first=True)
    packed = pack_padded_sequence(padded, lens, batch_first=True, enforce_sorted=False)
    return packed

def ckpt_save(ckpt_dir, epoch, model_params, optimizer_params, loss_params, loss_value, hp_params,
                latest_ckpt = False, best_val = False, early_stop = False):
    
    if latest_ckpt == True:
        ckpt_model_filename = 'latest_ckpt.pt'
    elif best_val == True:
        ckpt_model_filename = 'best_val.pt'
    elif early_stop == True:
        ckpt_model_filename = 'early_stop.pt'
    else:
        ckpt_model_filename = "ckpt_epoch_" + str(epoch) + ".pt"

    ckpt_model_path = os.path.join(ckpt_dir, ckpt_model_filename)
    ckpt_dict = {
            'epoch': epoch,
            'model_state_dict': model_params,
            'optimizer_state_dict': optimizer_params,
            'loss_state_dict': loss_params,
            'train_loss' : loss_value,
            'params'  : hp_params
            }
    torch.save(ckpt_dict, ckpt_model_path)
