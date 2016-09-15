# uniformly samples pixels in an image, regardless of class labels

import caffe
from config.options import options
import numpy as np
import numpy.random as npr
from util.blob import prep_im_for_blob_segdb, prep_gt_for_blob_segdb, im_list_to_blob_segdb
import cv2
import yaml

class segUniformLayer(caffe.Layer):

    def _shuffle(self):
        self._perm = npr.permutation(np.arange(self._len))
        self._cur = 0

    def _get_next_minibatch(self):
        if self._cur + options.seg.imbatch >= self._len:
            self._shuffle()

        inds = self._perm[self._cur:self._cur+options.seg.imbatch]
        self._cur += options.seg.imbatch
        ims = []
        pixels = np.zeros((0, 3), dtype=np.float32)
        labels = np.zeros((0), dtype=np.float32)

        im_flip = False
        for i, ii in enumerate(inds):
            # images
            im = cv2.imread(self._segdb.image_path_at(ii))
            if options.seg.trainflip:
                im_flip = npr.choice([True,False])
            # note that finetuning should change it, (test out)
            im_scale = npr.choice(options.seg.scales)
            ims.append(prep_im_for_blob_segdb(im, im_scale, im_flip, options.meanvalue))

            gt = cv2.imread(self._segdb.gt_path_at(ii),cv2.CV_LOAD_IMAGE_GRAYSCALE)
            gt = prep_gt_for_blob_segdb(gt,im_scale,im_flip)
            locs,labs = self._sample_pixels(gt, options.seg.samplesize)
            im_ind = i * np.ones((options.seg.samplesize, 1))
            pix = np.hstack((im_ind, locs))
            pixels = np.vstack((pixels,pix))

            labels = np.hstack((labels,labs))

        blob = im_list_to_blob_segdb(ims, self._pad, self._margin)
        return blob, pixels, labels

    def _sample_pixels(self, gt, samplesize):
        # (sample locations and get the labels)
        (y,x) = (gt<255.0).nonzero()
        v = gt[y,x]
        lv = len(v)
        c = np.arange(lv)
        if samplesize <= lv:
            inds = npr.choice(c, size=samplesize, replace=False)
        else:
            inds = npr.choice(c, size=samplesize, replace=True)
        y = y[inds]
        x = x[inds]
        labs = v[inds]
        locs = np.array([y,x]).transpose() + self._pad
        return locs, labs

    def set_segdb(self, segdb):
        self._segdb = segdb
        self._len = segdb.num_images
        # first shuffle
        self._shuffle()

    def setup(self, bottom, top):

        # In the most basic segmentation model, we do not need parameters..
        # number of classes is only needed when doing regression

        layer_params = yaml.load(self.param_str)
        self._pad = layer_params['pad']
        self._margin = self._pad + options.seg.margin

        top[0].reshape(1, 3, 224, 224)
        top[1].reshape(1000, 3)
        top[2].reshape(1000, 1)

    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass

    def forward(self, bottom, top):
        data, pixels, labels = self._get_next_minibatch()

        top[0].reshape(*(data.shape))
        top[0].data[...] = data.astype(np.float32, copy=False)

        top[1].reshape(*(pixels.shape))
        top[1].data[...] = pixels.astype(np.float32, copy=False)

        top[2].reshape(*(labels.shape))
        top[2].data[...] = labels.astype(np.float32, copy=False)