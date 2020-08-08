#!/usr/bin/env python

import common
from common import params, libraries
from common.libraries import *

import nnet
from nnet import model, dataset, loss, utils, visdomvisulization

def GetArgs():
    parser = argparse.ArgumentParser(description="Voice Activity Detection (webrtc)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in-data-dir', type=str, dest = "in_data_dir", required = True,
        help='Proivde valid input data directory')
    parser.add_argument('--ckpt-dir', type=str, dest = "ckpt_dir", required = True,
        help='Proivde valid checkpoint directory')
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
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    return args

def train_epoch(epoch):
    danet_model.train()
    iteration = 0
    train_loss = 0
    
    for batch_id, (batch_utt_id, batch_data) in enumerate(train_loader):
        batch_mixture_abs, batch_mixture_ph, batch_mixture_mel_stft, batch_ideal_mask, batch_weight_thresh = batch_data
        
        hidden = danet_model.init_hidden(hp.train.batch_size)
        optimizer.zero_grad()
        batch_est_mask, hidden = danet_model(hidden = hidden, batch_features = batch_mixture_mel_stft, batch_weight_thresh = batch_weight_thresh, batch_ideal_mask = batch_ideal_mask)
        loss = danet_loss(batch_features = batch_mixture_mel_stft, batch_ideal_mask = batch_ideal_mask, batch_est_mask = batch_est_mask)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(danet_model.parameters(), 3.0)

        train_loss = train_loss + loss.data.item()
        optimizer.step()
        
        if (batch_id + 1) % hp.train.log_interval == 0:
            mesg = "Epoch : {0}[{1}/{2}] | Iteration : {3} | Loss : {4:.4f} | Total Loss : {5:.4f} \n".format(
                        epoch+1, batch_id+1, len(train_loader), 
                        iteration, loss.data.item(), train_loss/(batch_id + 1))
            print(mesg)

            with open(log_file,'a') as f:
                f.write(mesg)

        iteration = iteration + 1
    
    train_loss = train_loss/len(train_loader)
    print('-' * 85)
    print('Epoch {0:3d} | Training Loss {1:.4f}'.format(epoch+1, train_loss))
    return train_loss

def do_training():
    trainLoss = []

    for epoch in range(hp.train.epochs):
        start_time = datetime.now()
        
        trainLoss.append(train_epoch(epoch))
    
        if hp.visdom.get_plot == True:
            train_vis.plot_loss(trainLoss[-1], epoch)

        #save danet_model
        danet_model.eval().cpu()

        if trainLoss[-1] == np.min(trainLoss):
            print('\tBest training model found.')
            utils.ckpt_save(ckpt_dir, epoch+1, danet_model.__str__(),
                            danet_model.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
                            danet_loss.state_dict(), trainLoss[-1], 0, dict(hp), best_train = True)

        end_time = datetime.now()
        print('Start Time : {0} | Elapsed Time: {1}'.format(str(start_time), str(end_time - start_time)))
        print('-' * 85)
        print('\n')
        if (epoch + 1) % hp.train.ckpt_interval == 0:
            utils.ckpt_save(ckpt_dir, epoch+1, danet_model.__str__(),
                            danet_model.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
                            danet_loss.state_dict(), trainLoss[-1], 0, dict(hp))

    utils.ckpt_save(ckpt_dir, epoch+1, danet_model.__str__(),
                            danet_model.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
                            danet_loss.state_dict(), trainLoss[-1], 0, dict(hp), latest_ckpt = True)

if __name__ == '__main__':
    args = GetArgs()
    hp = params.Hparam(args.conf_path).load_hparam()

    featsscp_path = os.path.join(args.in_data_dir, "feats.scp")
    if not os.path.exists(featsscp_path):
        raise Exception("This script expects" + featsscp_path + "to exist")

    ckpt_dir = args.ckpt_dir
    log_file = os.path.join(ckpt_dir, 'stats')

    with open(log_file, 'w') as f_log:
        f_log.truncate(0)

    if hp.visdom.get_plot == True:
        train_vis = visdomvisulization.visdomvisulization(env_name = hp.model_name, title = hp.model_name + "_" + hp.database + "_" + "train", xlabel = hp.visdom.xlabel, ylabel = hp.visdom.ylabel)

    danet_model = model.DANet(hp, "train").to(hp.device)
    danet_loss = loss.TFMaskLoss().to(hp.device)
    optimizer = optim.Adam(danet_model.parameters(), lr = hp.train.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    print(danet_model)
    
    train_data = dataset.MUSDB18Dataset(args.in_data_dir, hp)
    train_loader = DataLoader(train_data, batch_size = hp.train.batch_size, shuffle = True, drop_last = True, collate_fn = utils.load_train_batch)
    
    do_training()

    # if hp.train.restore:
    #     state_dict = torch.load(hp.model.model_path)
    #     danet_model.load_state_dict(state_dict['model_state_dict'])
    #     optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    #     danet_loss.load_state_dict(state_dict['loss_state_dict'])
    #     for state in optimizer.state.values():
    #         for k, v in state.items():
    #             if isinstance(v, torch.Tensor):
    #                 state[k] = v.cuda()
