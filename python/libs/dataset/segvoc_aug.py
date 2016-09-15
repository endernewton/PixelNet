import os.path as osp
import dataset
import dataset.segdb
# import dataset.seg
import scipy.io as sio
import cPickle as pickle
# import numpy as np

# script like class, everytime will ha
class segvoc_aug(dataset.segdb):
    def __init__(self,theset):
        name = 'aug-voc2012_' + theset
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
            vocimpath = '/scratch/xinleic/PASCAL/VOC2012/'
            vocpath = '/scratch/xinleic/PASCAL/dataset/'
            setpath = osp.join(vocpath,theset+'-aug.txt')
            with open(setpath) as fid:
                imagelist = fid.readlines()

            self._images = [vocimpath+'JPEGImages/'+im.strip()+'.jpg' for im in imagelist]
            self._gts = [vocpath+'PYsegCls/'+im.strip()+'.png' for im in imagelist]

            # get the classes
            # imdbpath = '/home/xinleic/GRA/External/imdb/imdb_voc_2012_'+theset+'.mat'
            # self._classes = sio.loadmat(imdbpath)['imdb'].ravel()
            # self._classes = [cls[0][0] for cls in self._classes]
            # self._classes.insert(0,'__background__')
            
            self._classes = ['__background__', # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor']

            with open(cache_file, 'wb') as fid:
                pickle.dump([self._images, self._gts, self._classes], fid)
