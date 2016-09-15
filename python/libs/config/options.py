# just following fast rcnn
import os
import os.path as osp
import numpy as np
import commands
from easydict import EasyDict as edict

options = edict()

# meta
options.homepath = osp.expanduser('~')
res = commands.getstatusoutput('nvidia-smi')
if not res[0]:
    options.machinename = 'gpu'
else:
    with open(osp.join(options.homepath,'machine.name')) as f:
        machinename = f.readlines()
        machinename = machinename[0].strip()
        options.machinename = machinename

# not sure if this will be printed somewhere
# print 'Machine Name: %s' % options.machinename

options.timelimit = float('inf')
if options.machinename == 'warp':
    options.timelimit = 22.0 * 3600
elif options.machinename == 'yoda':
    options.timelimit = 23.0 * 3600
elif options.machinename == 'workhorse':
    options.timelimit = 2.0 * 3600

options.meanvalue = np.array([[[102.9801, 115.9465, 122.7717]]])
options.seed = 1989
options.margin = 16
options.eps = 1e-14

# path
options.projectpath = osp.join(options.homepath,'BitP')
options.codepath = osp.join(options.projectpath,'Code')
options.netpath = osp.join(options.codepath,'net')
options.cachepath = osp.join(options.projectpath,'Cache')
options.cdatapath = osp.join(options.cachepath,'dataset')
options.cnetpath = osp.join(options.cachepath,'net')
options.disppath = osp.join(options.projectpath,'Disp')

# segment
options.seg = edict()

options.seg.datasets = ['context','context33','context20']
options.seg.valdatasets = ['context20','context20','context20']
options.seg.trainsets = ['train','train','train']
options.seg.testsets = ['val','val','val']
options.seg.num_classes =  [59, 33, 20]

options.seg.sizes = np.array([250])
options.seg.maxsize = 500
options.seg.margin = 16
options.seg.scales = options.seg.sizes.astype(np.float32) / options.seg.maxsize
options.seg.trainflip = True

# options.seg.train_net is computed
options.seg.base_lr = 0.001
options.seg.lr_policy = 'step'
options.seg.gamma = 0.1
# options.seg.stepsize is calculated
options.seg.display = 5
options.seg.average_loss = 100
options.seg.momentum = 0.9
options.seg.weight_decay = 0.0005
options.seg.snapshot = 0
options.seg.snapshot_prefix = 'notavailable'
options.seg.random_seed = 1989
options.seg.iter_size = 1

options.seg.imbatch = 5
options.seg.batchsize = 10000
options.seg.samplesize = options.seg.batchsize / options.seg.imbatch

options.seg.epoch = 80
options.seg.redepoch = 40
options.seg.saveepoch = 10
options.seg.fgrate = 0.5

options.seg.ftscaleopt = 'train'
options.seg.ftsizes = np.array([125,250,500])
options.seg.ftscales = options.seg.ftsizes.astype(np.float32) / options.seg.maxsize
options.seg.ftepoch = 8
options.seg.ftepochall = 24
options.seg.ftstartepoch = 10

options.seg.testsizes = np.array([500])
options.seg.testscales = options.seg.testsizes.astype(np.float32) / options.seg.maxsize
options.seg.testmaxsize = 500
options.seg.testscaleopt = 'train'
options.seg.testbatchsize = 16384
options.seg.testsamples = 0
options.seg.testdisp = 1
options.seg.testflip = True

options.seg.nets = ['vgg16-conv45fc7-nl2']
options.seg.netpads = [16]