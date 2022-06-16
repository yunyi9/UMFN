import torch
import torch.nn as nn
import torch.utils.data
import dataset as dsets
from torch.autograd import Variable
import module as net
import argparse
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--feapath', type=str, default='./data/VideoEmotion/tsnv3_rgb_pool/,./data/VideoEmotion/tsnv3_flow_pool,./data/VideoEmotion/audioset')
parser.add_argument('--dataset', type=str, default='VideoEmotion')
parser.add_argument('--numclasses', type=int, default=8)
parser.add_argument('--split', type=int, default=1)

parser.add_argument('--dtype', type=str, default='valence')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seqlen', type=int, default=50)
parser.add_argument('--hidden1', type=int, default=800)
parser.add_argument('--hidden2', type=int, default=800)
parser.add_argument('--numlayers', type=int, default=3)
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--numepochs', type=int, default=1)
parser.add_argument('--learate', type=float, default=0.0001)
parser.add_argument('--l2', type=float, default=0.01)
parser.add_argument('--ver', type=str, default='V1.0.0')
arg = parser.parse_args()
print(arg)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpu
seed = 1111
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

modelpath = '../outmodels'
if not os.path.exists(modelpath):
   os.mkdir(modelpath)


def calAcc(outputs, llabel):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == llabel).sum()
    total = len(llabel)
    ret = 100.0 * float(correct) / total
    return ret, correct, total


class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=255,
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        label[[a, torch.arange(label.size(1)), *b]] = 0
        if self.reduction == 'mean':
            loss = -torch.sum(torch.sum(logs*label, dim=1)) / n_valid
        elif self.reduction == 'none':
            loss = -torch.sum(logs*label, dim=1)
        return loss


def trainClassify(train_loader, module):
    print(module)
    module.train()
    criterion = LabelSmoothSoftmaxCE()
    optimizer = torch.optim.Adam(module.parameters(), lr=arg.learate, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=arg.l2, amsgrad=True)
    for epoch in range(arg.numepochs):
        closs = 0
        lpred, llable = [], []
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            outputs = module(images)
            loss = criterion(outputs, labels)
            closs += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 0.5)
            optimizer.step()
            llable.append(labels.cpu())
            lpred.append(outputs.data.cpu())
        tout = torch.cat(lpred, dim=0)
        tlab = torch.cat(llable, dim=0)
        ret, correct, total = calAcc(tout, tlab)

        print('Epoch [%d/%d], Step [%d], Loss: %.4f ACC:%.2f correct=%d, total=%d' %
              (epoch + 1, arg.numepochs, i + 1, closs / (i + 1), ret, correct, total))

    if arg.dataset == 'VideoEmotion':
        dicpath = '%s/%s_%d_%d.pt' % (modelpath, arg.dataset, arg.split, epoch + 1)
    else:
        dicpath = '%s/%s_%s_%d.pt' % (modelpath, arg.dataset, arg.dtype, epoch + 1)
    torch.save(module.state_dict(), dicpath)


def trainReg(train_loader, module):
    print(module)
    module.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(module.parameters(), lr=arg.learate, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=arg.l2, amsgrad=True)
    for epoch in range(arg.numepochs):
        closs = 0
        lpred, llable = [], []
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            outputs = module(images).squeeze(dim=1)
            mse = criterion(outputs, labels)
            loss = mse
            closs += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 0.5)
            optimizer.step()
            llable.append(labels.cpu())
            lpred.append(outputs.data.cpu())
        tout = torch.cat(lpred, dim=0).numpy()
        tlab = torch.cat(llable, dim=0).numpy()
        mse = mean_squared_error(tlab, tout)
        pcc, _ = pearsonr(tlab, tout)
        print('Epoch:[%d/%d] loss:%.4f MSE: %.3f, PCC: %.3f%%' %
              (epoch + 1, arg.numepochs, closs / (i + 1), mse, pcc*100))
    if arg.dataset == 'VideoEmotion':
        dicpath = '%s/%s_%d_%d.pt' % (modelpath, arg.dataset, arg.split, epoch + 1)
    else:
        dicpath = '%s/%s_%s_%d.pt' % (modelpath, arg.dataset, arg.dtype, epoch + 1)
    torch.save(module.state_dict(), dicpath)

def main():
    if 'EIMT16' == arg.dataset:
        train_dataset = dsets.EIMT16(arg.feapath, True, dimtype=arg.dtype, seqlen=arg.seqlen)
    elif 'VideoEmotion' == arg.dataset:
        train_dataset = dsets.VideoEmotion(arg.feapath, True, seqlen=arg.seqlen, split=arg.split, numclass=arg.numclasses)
    arg.feadim = train_dataset.feadim
    arg.numfeas = len(arg.feadim)
    print(arg)
    print(train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=arg.batchsize,
                                               shuffle=True, num_workers=3)
    module = net.UMFN(arg)
    module.cuda()
    if 'EIMT16' == arg.dataset:
        trainReg(train_loader, module)
    elif 'VideoEmotion' == arg.dataset:
        trainClassify(train_loader, module)


if __name__ == "__main__":
    main()

