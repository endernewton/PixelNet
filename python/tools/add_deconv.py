from __future__ import division
import sys
import os.path as osp
caffe_root = '/home/xinleic/BitP/Code/caffe'
sys.path.insert(0, osp.join(caffe_root, 'python'))
import argparse
import caffe
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Initialize all the deconvolution layers in a network')
    parser.add_argument('--define', dest='define',
                        help='network prototxt',
                        default=None, type=str)
    parser.add_argument('--caffemodel', dest='caffemodel',
                        help='network weights',
                        default=None, type=str)
    parser.add_argument('--filename', dest='filename',
                        help='file name to save the final network',
                        default=None, type=str)
    parser.add_argument('--seed', dest='seed',
                        help='seed for random stream',
                        default=1989, type=int)

    args = parser.parse_args()
    return args

# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    caffe.set_mode_cpu()
    caffe.set_random_seed(args.seed)
    net = caffe.Net(args.define, args.caffemodel, caffe.TEST)

    # do net surgery to set the deconvolution weights for bilinear interpolation
    interp_layers = [k for k in net.params.keys() if 'up' in k]
    interp_surgery(net, interp_layers)

    net.save(args.filename)

