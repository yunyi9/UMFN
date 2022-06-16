import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
import math


class UMFN(nn.Module):
    def __init__(self, arg):
        super(UMFN, self).__init__()
        self.num_layers = arg.numlayers
        self.feadim = arg.feadim
        self.numfeas = arg.numfeas
        self.dataset = arg.dataset
        self.modelfusion = UFL(arg.numfeas)
        self.modellist = nn.ModuleList()
        for i in range(self.numfeas):
            self.modellist.append(Stream(i, arg))

    def forward(self, x):
        lo = []
        dstart = 0
        for f in range(self.numfeas):
            dend = dstart + self.feadim[f]
            txd = x[:, :, dstart:dend]
            out = self.modellist[f](txd)
            dstart = dend
            lo.append(out)
        out = self.modelfusion(lo)
        return out

class UFL(nn.Module):
    def __init__(self, num):
        super(UFL, self).__init__()
        self.num = num
        self.w = nn.Parameter(torch.ones(self.num), requires_grad=True)

    def forward(self, x):
        if isinstance(x, list):
            out = torch.stack(x, dim=2)
            out = torch.mul(out, torch.softmax(self.w, dim=0))
            out = torch.sum(out, dim=2)
        else:
            out = torch.mul(x.permute(0, 2, 1), torch.softmax(self.w, dim=0))
            out = torch.sum(out, dim=2)
        return out


class Norm(nn.Module):
    def __init__(self, d_out):
        super(Norm, self).__init__()
        self.norm = nn.Sequential(nn.LayerNorm(d_out), nn.Dropout(0.1, inplace=True))

    def forward(self, x):
        return self.norm(x)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(RNN, self).__init__()
        self.rnn = nn.LSTMCell(input_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(0.1, inplace=True)
        self.d1 = input_size
        self.d2 = hidden_size
        self.h = None
        self.c = None
        self.resetparam()
        self.norm = nn.LayerNorm(hidden_size)

    def resetparam(self):
        init_weight = 0.1
        init.uniform_(self.rnn.weight_hh.data, -init_weight, init_weight)
        init.uniform_(self.rnn.weight_ih.data, -init_weight, init_weight)
        if True == self.rnn.bias:
            init.uniform_(self.rnn.bias_ih.data, -init_weight, init_weight)
            init.zeros_(self.rnn.bias_hh.data)

    def forward(self, x):
        bsz = x.shape[0]
        num = x.shape[1]
        if self.h is None:
            std = math.sqrt(2.0 / (self.d1 + self.d2))
            self.h = Tensor(bsz, self.d2).normal_(0, std)
            self.c = Tensor(bsz, self.d2).normal_(0, std)
        output = []
        h = self.h[:bsz, :]
        c = self.c[:bsz, :]
        for i in range(num):
            (h, c) = self.rnn(x[:, i, :], (h.cuda(), c.cuda()))
            output.append(h)
            h = self.dropout(h)
        self.h[:bsz, :] = h.detach()
        self.c[:bsz, :] = c.detach()
        output = torch.stack(output, dim=1)
        return self.norm(output)


class RRFB(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seqlen, d_out, numclasses):
        super(RRFB, self).__init__()
        self.num_layers = num_layers
        self.tempfusion = UFL(seqlen)
        self.layerfusion = UFL(num_layers)
        self.rnn = nn.ModuleList()
        self.rnn.append(RNN(input_size, hidden_size, bias=True))
        for l in range(self.num_layers - 1):
            self.rnn.append(RNN(hidden_size, hidden_size, bias=True))
        self.fc = nn.Sequential(nn.Linear(hidden_size, d_out), Norm(d_out), nn.Linear(d_out, numclasses))

    def forward(self, x):
        l1 = []
        out = self.rnn[0](x)
        l1.append(out)
        for l in range(1, self.num_layers):
            residual = out
            out = self.rnn[l](out)
            l1.append(out)
            out += residual
            out = F.relu(out, inplace=True)
        l2 = []
        for out in l1:
            out = self.fc(out)
            out = self.tempfusion(out)
            l2.append(out)
        out = self.layerfusion(l2)
        return out


class Stream(nn.Module):
    def __init__(self, i, arg):
        super(Stream, self).__init__()
        self.dirfusion = UFL(2)
        self.fc1 = nn.Sequential(nn.Linear(arg.feadim[i], arg.hidden1), Norm(arg.hidden1))
        self.dir1 = RRFB(arg.hidden1, arg.hidden2, arg.numlayers, arg.seqlen, 256, arg.numclasses)
        self.dir2 = RRFB(arg.hidden1, arg.hidden2, arg.numlayers, arg.seqlen, 256, arg.numclasses)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = x1.flip(1)
        x1 = self.dir1(x1)
        x2 = self.dir2(x2)
        x = self.dirfusion([x1, x2])
        return x
