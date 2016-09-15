import os.path as osp
import dataset
from config.options import options
import numpy as np
import cv2
# from util.timer import Timer
import cPickle as pickle
# import pdb

class segdb(object):
    def __init__(self,name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._images = []
        self._gts = []
        self._cachedir = options.cdatapath

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def images(self):
        return self._images

    @property
    def gts(self):
        return self._gts

    @property
    def cachepath(self):
        return osp.join(self._cachedir, self.name + '.pkl')

    @property
    def statspath(self):
        return osp.join(self._cachedir, self.name + '-stats.pkl')

    @property
    def num_images(self):
        return len(self._images)

    @property
    def num_classes(self):
        return len(self._classes)

    def image_path_at(self, i):
        return self._images[i]

    def gt_path_at(self, i):
        return self._gts[i]

    def compute_stats(self):
        cache_file = self.statspath

        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                stats, dists = pickle.load(fid)
            print 'stats for {} classes loaded from {}'.format(self.num_classes, \
                cache_file)
            return stats, dists

        bins = np.arange(self.num_classes + 1)
        stats = np.zeros(self.num_classes, dtype=np.uint64)

        # timer = Timer()
        # timer.tic()
        for i in np.arange(self.num_images):
            gt = cv2.imread(self.gt_path_at(i),cv2.CV_LOAD_IMAGE_GRAYSCALE)
            gt = gt.astype(np.uint8, copy=False)
            ind = (gt<255).nonzero()
            gt = gt[ind]
            stat, _ = np.histogram(gt.flat,bins)
            stats += stat.astype(np.uint64, copy=False)
            # timer.toc()

        print stats
        dists = stats.astype(np.float64, copy=False)
        dists = dists / dists.sum()
        print dists

        with open(cache_file, 'wb') as fid:
            pickle.dump([stats, dists], fid)

        return stats, dists