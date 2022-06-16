import torch.utils.data as data
import numpy as np
import os
import os.path


def norm2dl2(dat):
    norm = np.sqrt(np.sum(dat*dat,axis=1)) + 1e-30
    dt = np.transpose(np.transpose(dat)/norm)
    return dt


class DataBase(data.DataLoader):
    lstrain = ''
    lstest = ''

    def _readlist_(self, listf, dimtype='arousal'):
        pass

    def __init__(self, root, train=True, dimtype='arousal',seqlen=160):
        self.root = root.split(',')
        self.train = train  # training set or test set
        self.vid = []
        self.label = []
        self.seqlen = seqlen
        if not self._check_exists():
            raise RuntimeError('Dataset not found.')
        if self.train:
            self.vid, self.label = self._readlist_(self.lstrain, dimtype)
        else:
            self.vid, self.label = self._readlist_(self.lstest, dimtype)
        self.feadim = []
        for r in self.root:
            fpath = '%s/%s.npy' % (r, self.vid[0])
            dim = np.load(fpath).shape[1]
            if r.find('audioset') == -1:
                self.feadim.append(dim)
                self.feadim.append(dim)
            else:
                self.feadim.append(dim)

    def __getitem__(self, index):
        lvfea = []
        for r in self.root:
            target = self.label[index]
            fpath = '%s/%s.npy'%(r,self.vid[index])
            vdata = np.load(fpath)
            nv = vdata.shape[0]
            span = 25
            if r.find('audioset') == -1:
                maxlen = self.seqlen * span
            else:
                maxlen = self.seqlen
            if nv < maxlen:
                while vdata.shape[0] < maxlen:
                    vdata = np.vstack((vdata, vdata))
            nv = vdata.shape[0]
            if r.find('audioset') == -1:
                dsize = vdata.shape[1]
                vfea1 = np.zeros((self.seqlen, dsize), dtype=vdata.dtype)
                vfea2 = np.zeros((self.seqlen, dsize), dtype=vdata.dtype)
                for i in range(self.seqlen):
                    tmp = vdata[i*span:(i+1)*span,:]
                    dtm = np.mean(tmp,axis=0)
                    dts = np.std(tmp,axis=0)
                    vfea1[i, :] = dtm
                    vfea2[i, :] = dts / (dtm + 1e-30)
                vfea = np.float32(np.hstack([norm2dl2(vfea1),norm2dl2(vfea2)]))
            else:
                vfea = np.float32(norm2dl2(vdata[:self.seqlen,:]))
            lvfea.append(vfea)
        fea = np.hstack(lvfea)
        return fea, target

    def __len__(self):
        return len(self.label)

    def _check_exists(self):
        return os.path.exists(self.root[0])


class EIMT16(DataBase):
    lstrain = './list/EIMT16_train.list'
    lstest = './list/EIMT16_test.list'

    def _readlist_(self, listf, dimtype='arousal'):
        vid = []
        label = []
        with open(listf,'r') as lfo:
            print('load list ...%s'%listf)
            lines = lfo.readlines()
            for l in lines:
                #SHOT_000000 1 -1 id valence arousal
                line = l.split(' ')
                if len(line) != 3:
                    print('[ERROR] [%s], line [%s]  error'%(listf, l))
                    exit(0)
                vid.append(line[0].split('/')[-1])

                if dimtype == 'arousal':
                    clabel= np.float32(line[2])
                elif dimtype == 'valence':
                    clabel= np.float32(line[1])
                else:
                    exit(0)

                label.append(clabel)
        return vid, label

    def __init__(self, root, train=True, dimtype='arousal',seqlen=160):
        super().__init__(root, train, dimtype,seqlen)


class VideoEmotion(DataBase):
    def _readlist_(self, listf, dimtype=''):
        vid = []
        label = []
        with open(listf,'r') as lfo:
            print('load list ...%s'%listf)
            lines = lfo.readlines()
            for l in lines:
                line = l.split(' ')
                if len(line) != 2:
                    print('[ERROR] [%s], line [%s]  error'%(listf, l))
                    exit(0)
                vid.append(line[0].split('/')[-1])
                clabel = int(line[1])
                label.append(clabel)
        return vid, label

    def __init__(self, root, train=True,seqlen=160, split=1, numclass=8):
        if 8 == numclass:
            self.lstrain = './list/VE_%d_Trainset.list'%(split)
            self.lstest = './list/VE_%d_Testset.list'%(split)
        else:
            self.lstrain = './list/VE4_%d_Trainset.list'%(split)
            self.lstest = './list/VE4_%d_Testset.list'%(split)
        super().__init__(root, train, 'arousal',seqlen)
