#!/usr/bin/env python

from . import libraries, params
from .libraries import *

class TFMaskLoss(nn.Module):
    def __init__(self):
        super(TFMaskLoss, self).__init__()

    def forward(self, batch_features, batch_ideal_mask, batch_est_mask):
        # batch_features : [B, T, F])
        # batch_ideal_mask : [B, T*F, nspk]
        
        batch_features = batch_features.view(batch_features.size(0), batch_features.size(1) * batch_features.size(2),1)
        # batch_features : [B, T*F, 1])

        batch_features = batch_features.expand_as(batch_ideal_mask)
        # batch_features : [B, T*F, nspk])

        batch_loss = batch_features * (batch_ideal_mask - batch_est_mask)
        # batch_loss : [B, T*F, nspk])

        batch_loss = batch_loss.view(-1,batch_loss.size(1)*batch_loss.size(2))
        # batch_loss : [B, T*F*nspk])

        return torch.mean(torch.sum(torch.pow(batch_loss, 2), 1))
