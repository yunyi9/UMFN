import torch
import cv2
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

# parser.add_argument('--feapath', type=str, default='./data/EIMT16/me16tsn_v3_rgb,./data/EIMT16/me16tsn_v3_flow,./data/EIMT16/audioset')
# parser.add_argument('--dataset', type=str, default='EIMT16')
# parser.add_argument('--numclasses', type=int, default=1)

parser.add_argument('--dtype', type=str, default='arousal')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--numepochs', type=int, default=5)

parser.add_argument('--seqlen', type=int, default=50)
parser.add_argument('--hidden1', type=int, default=800)
parser.add_argument('--hidden2', type=int, default=800)
parser.add_argument('--numlayers', type=int, default=3)
parser.add_argument('--ver', type=str, default='V1.0.0')
arg = parser.parse_args()


import os
os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpu
seed = 1111
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


def pearson_correlation(x1, x2, eps=1e-8):
    assert x1.dim() == 1, "Input must be 1D matrix / vector."
    assert x1.size() == x2.size(), "Input sizes must be equal."
    x1_bar = x1 - x1.mean()
    x2_bar = x2 - x2.mean()
    dot_prod = x1_bar.dot(x2_bar)
    norm_prod = x1_bar.norm(2) * x2_bar.norm(2)
    return dot_prod / norm_prod.clamp(min=eps)


def calAcc(outputs, llabel):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == llabel).sum()
    total = len(llabel)
    ret = 100.0 * float(correct) / total
    return ret, correct, total


def testClassify(test_loader, ld, arg):
    with torch.no_grad():
        module = net.UMFN(arg)
        module.cuda()
        module.eval()
        dict = torch.load(ld)
        module.load_state_dict(dict)
        lpred, llable = [], []
        for i, (images, labels) in enumerate(test_loader):
            images = Variable(images).cuda()
            outputs = module(images)
            llable.append(labels.cpu())
            lpred.append(outputs.data.cpu())
        tout = torch.cat(lpred, dim=0)
        tlab = torch.cat(llable, dim=0)
        ret, correct, total = calAcc(tout, tlab)
    return ret


def testReg(test_loader, ld, arg):
    with torch.no_grad():
        module = net.UMFN(arg)
        module.cuda()
        module.eval()
        dict = torch.load(ld)
        module.load_state_dict(dict)
        lpred, llable = [], []
        for i, (images, labels) in enumerate(test_loader):
            images = Variable(images).cuda()
            outputs = module(images)
            llable.append(labels.cpu())
            lpred.append(outputs.data.cpu())
        tout = torch.cat(lpred, dim=0).numpy()
        tlab = torch.cat(llable, dim=0).numpy()
        sout = cv2.GaussianBlur(np.array(tout), (1, 7), 0)
        mse = mean_squared_error(tlab, sout)
        pcc, _ = pearsonr(tlab, sout)
    return mse, pcc


def main():
    modelpath = '../outmodels'
    if 'EIMT16' == arg.dataset:
        test_dataset = dsets.EIMT16(arg.feapath, False, dimtype=arg.dtype, seqlen=arg.seqlen)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=3)
        arg.feadim = test_dataset.feadim
        arg.numfeas = len(arg.feadim)
        path = '%s/%s_%s_%d.pt' % (modelpath, arg.dataset, arg.dtype, arg.numepochs)
        mse, pcc = testReg(test_loader, path, arg)
        print(path)
        print('EIMT16 %s MSE=%.3f PCC=%.3f' % (arg.dtype, mse, pcc))
    elif 'VideoEmotion' == arg.dataset:
        lacc = []
        for split in range(1, 11):
            test_dataset = dsets.VideoEmotion(arg.feapath, False, seqlen=arg.seqlen, split=split, numclass=arg.numclasses)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, num_workers=3)
            arg.feadim = test_dataset.feadim
            arg.numfeas = len(arg.feadim)
            path = '%s/%s_%d_%d.pt' % (modelpath, arg.dataset, split, arg.numepochs)
            print(path)
            ret = testClassify(test_loader, path, arg)
            lacc.append(ret)
        print(lacc)
        print('VideoEmotion Mean ACC=%.3f' % (np.mean(lacc)))


if __name__ == "__main__":
    main()

