import os.path as osp
import dataset
import dataset.segdb
# import dataset.segpascalcontext
import scipy.io as sio
import cPickle as pickle
# import numpy as np

# script like class, everytime will ha
class segpascalcontext(dataset.segdb):
    def __init__(self,theset):
        name = 'context_' + theset
        dataset.segdb.__init__(self,name)
        cache_file = self.cachepath

        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                self._images, self._gts, self._classes = pickle.load(fid)
            print '{}: {} images/{} classes loaded from {}'.format(self.name, \
                self.num_images,self.num_classes,\
                cache_file)
        else:
            # get images and gts
            vocpath = '/scratch/xinleic/PASCAL/VOC2010/'
            labelpath = '/scratch/xinleic/PASCAL/PASCALContext/py-trainval-59/'
            setpath = osp.join(vocpath,'ImageSets','Main',theset+'.txt')
            with open(setpath) as fid:
                imagelist = fid.readlines()

            self._images = [vocpath+'JPEGImages/'+im.strip()+'.jpg' for im in imagelist]
            self._gts = [labelpath+im.strip()+'.png' for im in imagelist]

            # get the classes
            clsfile = '/scratch/xinleic/PASCAL/PASCALContext/clses-59.mat'
            self._classes = sio.loadmat(clsfile)['clses'].ravel()
            self._classes = [cls[0][0] for cls in self._classes]
            self._classes.insert(0,'__background__')

            with open(cache_file, 'wb') as fid:
                pickle.dump([self._images, self._gts, self._classes], fid)

# script like class, everytime will ha
class segpascalcontext20(dataset.segdb):
    def __init__(self,theset):
        name = 'context20_' + theset
        dataset.segdb.__init__(self,name)
        cache_file = self.cachepath

        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                self._images, self._gts, self._classes = pickle.load(fid)
            print '{}: {} images/{} classes loaded from {}'.format(self.name, \
                self.num_images,self.num_classes,\
                cache_file)
        else:
            # get images and gts
            vocpath = '/scratch/xinleic/PASCAL/VOC2010/'
            labelpath = '/scratch/xinleic/PASCAL/PASCALContext/py-trainval-20/'
            setpath = osp.join(vocpath,'ImageSets','Main',theset+'.txt')
            with open(setpath) as fid:
                imagelist = fid.readlines()

            self._images = [vocpath+'JPEGImages/'+im.strip()+'.jpg' for im in imagelist]
            self._gts = [labelpath+im.strip()+'.png' for im in imagelist]

            # get the classes
            clsfile = '/scratch/xinleic/PASCAL/PASCALContext/clses-20.mat'
            self._classes = sio.loadmat(clsfile)['clses'].ravel()
            self._classes = [cls[0][0] for cls in self._classes]
            self._classes.insert(0,'__background__')

            with open(cache_file, 'wb') as fid:
                pickle.dump([self._images, self._gts, self._classes], fid)

# script like class, everytime will ha
class segpascalcontext33(dataset.segdb):
    def __init__(self,theset):
        name = 'context33_' + theset
        dataset.segdb.__init__(self,name)
        cache_file = self.cachepath

        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                self._images, self._gts, self._classes = pickle.load(fid)
            print '{}: {} images/{} classes loaded from {}'.format(self.name, \
                self.num_images,self.num_classes,\
                cache_file)
        else:
            # get images and gts
            vocpath = '/scratch/xinleic/PASCAL/VOC2010/'
            labelpath = '/scratch/xinleic/PASCAL/PASCALContext/py-trainval-33/'
            setpath = osp.join(vocpath,'ImageSets','Main',theset+'.txt')
            with open(setpath) as fid:
                imagelist = fid.readlines()

            self._images = [vocpath+'JPEGImages/'+im.strip()+'.jpg' for im in imagelist]
            self._gts = [labelpath+im.strip()+'.png' for im in imagelist]

            # get the classes
            clsfile = '/scratch/xinleic/PASCAL/PASCALContext/clses-33.mat'
            self._classes = sio.loadmat(clsfile)['clses'].ravel()
            self._classes = [cls[0][0] for cls in self._classes]
            self._classes.insert(0,'__background__')

            with open(cache_file, 'wb') as fid:
                pickle.dump([self._images, self._gts, self._classes], fid)
