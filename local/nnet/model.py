#!/usr/bin/env python

from . import libraries, params
from .libraries import *

class MultiRNN(nn.Module):
    def __init__(self, hp):
        super(MultiRNN,self).__init__()
        self.hp = hp
        self.num_directions = int(self.hp.model.multirnn.bidirectional) + 1
        
        self.rnn = nn.LSTM(input_size = self.hp.model.multirnn.input_size, 
                           hidden_size = self.hp.model.multirnn.hidden_size, 
                           dropout = self.hp.model.multirnn.dropout, 
                           bidirectional = self.hp.model.multirnn.bidirectional,
                           num_layers = self.hp.model.multirnn.num_layers, 
                           batch_first = True)
        
    def forward(self,input,hidden):
        self.rnn.flatten_parameters()
        return self.rnn(input, hidden)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.hp.model.multirnn.num_layers* self.num_directions, batch_size, self.hp.model.multirnn.hidden_size).zero_().to(self.hp.device)),
                Variable(weight.new(self.hp.model.multirnn.num_layers* self.num_directions, batch_size, self.hp.model.multirnn.hidden_size).zero_().to(self.hp.device)))

class LinearLayer(nn.Module):
    def __init__(self, hp):
        super(LinearLayer,self).__init__()
        self.hp = hp
        self.linear = nn.Linear(in_features = self.hp.model.fc.input_size, out_features = self.hp.model.fc.output_size)
        self.init_hidden()
        
    def forward(self,data):
            return torch.tanh(self.linear(data))
        
    def init_hidden(self):
        initrange = 1. / np.sqrt(1/self.hp.model.fc.input_size) 
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-initrange,initrange)

class DANet(nn.Module):
    def __init__(self, hp):
        super(DANet,self).__init__()
        self.hp = hp

        self.rnn = MultiRNN(self.hp)
        self.linear = LinearLayer(self.hp)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch_features, hidden, batch_weight_thresh, batch_ideal_mask):
        # input : [B, T, F]
        # batch_ibm : [B, T*F, nspk]
        # batch_weight_thresh : [B, T*F, 1]
        
        batch_features = batch_features.to(self.hp.device)
        
        self.rnn.to(self.hp.device)

        LSTMOutput, hidden = self.rnn(batch_features, hidden)
        # LSTMOutput : [B, T, numDirection*hiddenSize]
        # hidden : [numDirections*numLayers, B, hiddenSize]

        LSTMOutput = LSTMOutput.contiguous().view(-1,LSTMOutput.size(2)) 
        # LSTMOutput : [B*T, noOfDirections*hiddenSize]
        self.linear.to(self.hp.device)

        V = self.linear(LSTMOutput) 

        V = self.tanh(V)
        # V : [B*T,F*K]

        V = V.view(-1, self.hp.model.sequence_length * self.hp.model.feature_size, self.hp.model.embedding_size) 
        # V : [B, T*F, K]

        # batch_weight_thresh = batch_weight_thresh.view(-1, self.hp.model.sequence_length * self.hp.model.feature_size, 1)

        # Y = batch_ibm * batch_weight_thresh.expand_as(batch_ibm)
        Y = batch_ideal_mask
        # Y : [B, T*F, nspk]

        V_Y = torch.bmm(torch.transpose(V, 1,2), Y) 
        # torch.transpose(V, 1,2) : [B, K, T*F]
        # V_Y : [B, K, nspk]

        sum_Y = torch.sum(Y, 1, keepdim=True)
        # sum_Y : [B, 1, nspk]

        sum_Y = sum_Y.expand_as(V_Y) 
        # sum_Y : [B, K, nspk]

        attractor = V_Y / (sum_Y + float(self.hp.eps))
        # attractor : [B, K, nspk]

        dist = V.bmm(attractor) 
        # dist : [B, T*F, nspk]

        mask = self.sigmoid(dist) 
        # mask : [B, T*F, nspk]
        
        return mask, hidden

    def init_hidden(self, batch_size):
        return self.rnn.init_hidden(batch_size)


class DANet_test(nn.Module):
    def __init__(self):
        super(DANet_test,self).__init__()
        self.hp = params.Hparam().load_hparam()
        self.eps = 1e-8

        self.rnn = model_base.MultiRNN()

        self.linear = model_base.FCLayer()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden, batch_features, batch_weight_thresh, batch_ideal_mask):
        # input : [B, T, F]
        # batch_ibm : [B, T*F, nspk]
        # batch_weight_thresh : [B, T*F, 1]
        batch_features = batch_features.to(self.hp.device)

        self.rnn.to(self.hp.device)
        LSTMOutput, hidden = self.rnn(batch_features, hidden)
        # LSTMOutput : [B, T, numDirection*hiddenSize]
        # hidden : [numDirections*numLayers, B, hiddenSize]

        LSTMOutput = LSTMOutput.contiguous().view(-1,LSTMOutput.size(2)) 
        # LSTMOutput : [B*T, noOfDirections*hiddenSize]

        self.linear.to(self.hp.device)

        V = self.linear(LSTMOutput) 

        V = self.tanh(V)
        # V : [B*T,F*K]

        V = V.view(self.hp.test.batch_size, -1 , self.hp.model.embedding_size)
        # V = V.view(-1, self.hp.model.sequence_length * self.hp.model.feature_size, self.hp.model.embedding_size) 
        # V : [B, T*F, K]

        return V

    def init_hidden(self, batch_size):
        return self.rnn.init_hidden(batch_size)