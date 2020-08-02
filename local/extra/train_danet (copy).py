#!/usr/bin/env python3

import os
import sys
import time
import h5py
import librosa
import argparse
import numpy as np

import nnet
from nnet import params, model, utils, loss, visdomvisulization

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

def train_epoch(train_loader, epoch):
    danet_model.train()
    iteration = 0
    train_loss = 0
    
    for batch_id, (batch_data, _ ) in enumerate(train_loader):
        batch_mix_features, batch_mix_abs, batch_mix_ph, batch_weight_thresh, batch_ideal_mask = utils.load_batch(batch_data, hp.training)
        hidden = danet_model.init_hidden(hp.train.batch_size)
        optimizer.zero_grad()
        batch_est_mask, hidden = danet_model(hidden = hidden, batch_input_features = batch_mix_features,
                                            batch_weight_thresh = batch_weight_thresh, batch_ideal_mask = batch_ideal_mask)

        loss = compute_loss(batch_mix_features = batch_mix_features, batch_ideal_mask = batch_ideal_mask, batch_est_mask = batch_est_mask)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(danet_model.parameters(), 3.0)
        # for data in danet_model.parameters():
        #     print(data)

        train_loss = train_loss + loss.data.item()
        optimizer.step()
        
        if (batch_id + 1) % hp.train.log_interval == 0:
            mesg = "Epoch : {0}[{1}/{2}] | Iteration : {3} | Loss : {4:.4f} | Total Loss : {5:.4f} \n".format(
                        epoch+1, batch_id+1, len(train_loader), 
                        iteration, loss.data.item(), train_loss/(batch_id + 1))
            print(mesg)

            with open(os.path.join(hp.train.ckpt_dir,'stats'),'a') as f:
                f.write(mesg)

            # if hp.train.log_file is not None:
            #     with open(hp.train.log_file,'a') as f:
            #         f.write(mesg)
        iteration = iteration + 1
    
    train_loss = train_loss/len(train_loader)
    print('-' * 85)
    print('Epoch {0:3d} | Training Loss {1:.4f}'.format(epoch+1, train_loss))
    return train_loss

def validation_epoch(validation_loader, epoch):
    
    danet_model.eval()
    val_loss = 0

    for batch_id, (batch_data, _ ) in enumerate(validation_loader):
        batch_mix_features, batch_mix_abs, batch_mix_ph, batch_weight_thresh, batch_ideal_mask  = utils.load_batch(batch_data, hp.training)

        with torch.no_grad():
            hidden = danet_model.init_hidden(hp.train.batch_size)

            batch_est_mask, hidden = danet_model(hidden = hidden, batch_input_features = batch_mix_features, 
                                                   batch_weight_thresh = batch_weight_thresh, batch_ideal_mask = batch_ideal_mask)
            
            loss = compute_loss(batch_mix_features = batch_mix_features, batch_ideal_mask = batch_ideal_mask, batch_est_mask = batch_est_mask)
            val_loss += loss.data.item()

    val_loss /= len(validation_loader)
    print('Epoch {0:3d} | Validation Loss {1:.4f}'.format(epoch+1, val_loss))

    return val_loss

def do_training():
    trainLoss = []
    valLoss = []
    decayCount = 0
    
    start_epoch = 0
    end_epoch = hp.train.epochs
    
    if hp.train.restore:
        start_epoch = state_dict['epoch']

    for epoch in range(start_epoch, end_epoch):
        start_time = datetime.now()
        
        trainLoss.append(train_epoch(train_loader, epoch))
        valLoss.append(validation_epoch(validation_loader, epoch))
        
        train_vis.plot_loss(trainLoss[-1], epoch, "Train")
        val_vis.plot_loss(valLoss[-1], epoch, "Validation")


        if trainLoss[-1] == np.min(trainLoss):
            print('\tBest training model found.')
        if valLoss[-1] == np.min(valLoss):
            print('\tBest validation model found')
            # save current best model
            utils.ckpt_save(hp.train.ckpt_dir, end_epoch, 
                            danet_model.state_dict(), optimizer.state_dict(),
                            compute_loss.state_dict(), trainLoss[-1], dict(hp), best_val = True)

        decayCount += 1

        # lr decay
        if np.min(valLoss) not in valLoss[-3:] and decayCount >= 3:
            scheduler.step()
            decayCount = 0
            print('\tLearning rate decreased.')

        if epoch >=10:
            if all(valLoss[-1] > x for x in valLoss[-10:]):
                print('\tEarly Stopping Satisfied')
                utils.ckpt_save(hp.train.ckpt_dir, end_epoch, 
                            danet_model.state_dict(), optimizer.state_dict(),
                            compute_loss.state_dict(), trainLoss[-1], dict(hp), early_stop = True)

        end_time = datetime.now()
        print('Start Time : {0} | Elapsed Time: {1}'.format(str(start_time), str(end_time - start_time)))
        print('-' * 85)
        print('\n')
        if hp.train.ckpt_dir is not None and (epoch + 1) % hp.train.ckpt_interval == 0:
            utils.ckpt_save(hp.train.ckpt_dir, epoch+1, 
                                    danet_model.state_dict(), optimizer.state_dict(),
                                    compute_loss.state_dict(), trainLoss[-1], dict(hp))
            
    #save danet_model
    danet_model.eval().cpu()
    utils.ckpt_save(hp.train.ckpt_dir, end_epoch, 
                            danet_model.state_dict(), optimizer.state_dict(),
                            compute_loss.state_dict(), trainLoss[-1], dict(hp), latest_ckpt = True)

def main():
    args = GetArgs()
    hp = params.Hparam(args.conf_path).load_hparam()

    featsscp_path = os.path.join(args.in_data_dir, "feats.scp")
    if not os.path.exists(featsscp_path):
        raise Exception("This script expects" + featsscp_path + "to exist")

    compute_loss = loss.TFMaskLoss().to(hp.device)

    train_vis = visdomvisulization.visdomvisulization()
    val_vis = visdomvisulization.visdomvisulization()

    danet_model = model.DANet().to(hp.device)

    optimizer = optim.Adam(danet_model.parameters(), lr = hp.train.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    train_data = dataset.MUSDB18Dataset()
    train_loader = DataLoader(train_data, batch_size = hp.train.batch_size, shuffle = True, drop_last = True)
    
    validation_data = dataset.MUSDB18Dataset(os.path.join(hp.data.dataset_path, hp.data.feature_folder, hp.train.seq_type, 'val','*/*.npz'))
    validation_loader = DataLoader(validation_data, batch_size = hp.train.batch_size, shuffle = True, drop_last = True)

    do_training()

if __name__ == '__main__':
    main()


    # if hp.train.restore:
    #     state_dict = torch.load(hp.model.model_path)
    #     danet_model.load_state_dict(state_dict['model_state_dict'])
    #     optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    #     compute_loss.load_state_dict(state_dict['loss_state_dict'])
    #     for state in optimizer.state.values():
    #         for k, v in state.items():
    #             if isinstance(v, torch.Tensor):
    #                 state[k] = v.cuda()
