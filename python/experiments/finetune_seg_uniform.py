import _init_paths
import caffe
from config.options import options
import seg_uniform.layer as layer
from dataset.factory import get_imdb
from net.initial_network import initial_network
import time
from util.timer import Timer
import numpy as np
import os
import os.path as osp
import sys
import pprint
import argparse
import commands
# import pdb

from caffe.proto import caffe_pb2
import google.protobuf as pb2

class SolverWrapper(object):

    def __init__(self, solverpath, segdb, trainfolder, tag, initmodelpath):
        self.epochfolder = osp.join(trainfolder, tag)
        self.finalfile = osp.join(trainfolder,tag+'.caffemodel')
        if not osp.isdir(self.epochfolder):
            os.makedirs(self.epochfolder)
        self.solver = caffe.SGDSolver(solverpath)
        self.solver.net.copy_from(initmodelpath)

        self.snapshotiter = 1

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solverpath, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_segdb(segdb)
        self.num_images = segdb.num_images

    def __del__(self):
        self.solver = None

    def snapshot(self):
        if self.snapshotiter == options.seg.ftepochall:
            filename = self.finalfile
        else:
            netname = '(%02d).caffemodel' % self.snapshotiter
            filename = osp.join(self.epochfolder, netname)

        if not osp.exists(filename):
            self.solver.net.save(filename)

    def train_model(self):
        while self.snapshotiter <= options.seg.ftepochall:
            for i in np.arange(0,self.num_images,options.seg.imbatch):
                self.solver.step(1)

                # just for the sake of display
                if np.random.uniform() > 0.9:
                    print self.epochfolder

            if self.snapshotiter % options.seg.saveepoch == 0:
                self.snapshot()
            self.snapshotiter += 1

        self.snapshotiter -= 1
        self.snapshot()

def parse_args():
    parser = argparse.ArgumentParser(description='finetune, segmentation, uniform')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def dump_prototxts(dataset, trainset, num_classes, num_images, net, pad):
    savefolder = osp.join(options.cnetpath,net)
    if not osp.isdir(savefolder):
        os.makedirs(savefolder)
    
    train_prototxt = osp.join(savefolder, 'train-uniform-' + dataset + '-' + trainset + '.prototxt')
    if not osp.exists(train_prototxt):
        with open(train_prototxt,'w') as f:
            f.write("name: '%s'\n" % net)
            f.write("layer {\n  name: 'data'\n  type: 'Python'\n")
            f.write("  top: 'data'\n")
            f.write("  top: 'pixels'\n")
            f.write("  top: 'labels'\n")
            f.write("  python_param {\n")
            f.write("    module: 'seg_uniform.layer'\n")
            f.write("    layer: 'segUniformLayer'\n")
            f.write("    param_str: \"'pad': %d\"\n" % pad)
            f.write("  }\n")
            f.write("}\n")
        tailfile = osp.join(options.netpath,net,'tail-train-%d.prototxt' % num_classes)
        commands.getoutput('cat %s >> %s' % (tailfile, train_prototxt))
    print 'Train-prototxt:', train_prototxt

    solver_prototxt = osp.join(savefolder, 'solver-finetune-uniform-' + dataset + '-' + trainset + '.prototxt')
    if not osp.exists(solver_prototxt):
        with open(solver_prototxt,'w') as f:
            f.write("train_net: '%s'\n" % train_prototxt)
            f.write("base_lr: %f\n" % options.seg.base_lr)
            f.write("lr_policy: '%s'\n" % options.seg.lr_policy)
            f.write("gamma: %f\n" % options.seg.gamma)
            f.write("stepsize: %d\n" % (num_images / options.seg.imbatch * options.seg.ftepoch))
            f.write("display: %d\n" % options.seg.display)
            f.write("average_loss: %d\n" % options.seg.average_loss)
            f.write("momentum: %f\n" % options.seg.momentum)
            f.write("weight_decay: %f\n"  % options.seg.weight_decay)
            f.write("snapshot: %d\n" % options.seg.snapshot)
            f.write("snapshot_prefix: '%s'\n" % options.seg.snapshot_prefix)
            f.write("random_seed: %d\n" % options.seg.random_seed)
            f.write("iter_size: %d\n" % options.seg.iter_size)

    print 'Solver-prototxt:', solver_prototxt
    return solver_prototxt, train_prototxt

def train_model(dataset, trainset, num_classes, net, pad, cachepath):
    cachefolder = osp.join(cachepath, dataset+'_'+trainset, net)
    if not osp.isdir(cachefolder):
        os.makedirs(cachefolder)
    ptag = 'S' + ('%d_'*len(options.seg.sizes)) % tuple(options.seg.sizes) \
             + 'IB%d_B%d_E%d-uniform' % (options.seg.imbatch,options.seg.batchsize,options.seg.epoch)
    if options.seg.trainflip:
        ptag += '_F'

    tag = 'S' + ('%d_'*len(options.seg.sizes)) % tuple(options.seg.sizes) \
             + 'IB%d_B%d_E%d-uniform' % (options.seg.imbatch,options.seg.batchsize,options.seg.ftepochall)
    if options.seg.trainflip:
        tag += '_F'

    trainfolder = osp.join(cachefolder,'TRAIN')
    finetunefolder = osp.join(cachefolder,'FT-TR')
    if not osp.isdir(finetunefolder):
        os.makedirs(finetunefolder)

    prefile = osp.join(trainfolder,ptag+'.caffemodel')
    targetfile = osp.join(finetunefolder,tag+'.caffemodel')
    targetlock = osp.join(finetunefolder,tag+'.lock')
    if osp.exists(targetfile) or not osp.exists(prefile):
        return
    try:
        os.mkdir(targetlock)
    except Exception as e:
        return
    # looks like it is a tricky task to redirect the stdout/stderr
    # leave it as of now
    # logfolder = osp.join(trainfolder,'logs')
    print '%s_%s<-%s: %s' % (dataset,trainset,net,tag)
    time.sleep(5)
    datasetname = '%s_%s' % (dataset,trainset)
    segdb = get_imdb(datasetname)

    # create the solver and train file
    solverpath, _ = dump_prototxts(dataset, trainset, num_classes, segdb.num_images, net, pad)
    # solverpath = osp.join(options.netpath,net,'pysol-seg-'+dataset+'-'+trainset+'.prototxt')

    # start training
    # caffe.set_random_seed(options.seed)
    np.random.seed(options.seed)
    sw = SolverWrapper(solverpath, segdb, finetunefolder, tag, prefile)
    sw.train_model()
    del sw

    # after training
    os.rmdir(targetlock)

if __name__ == '__main__':
    timer = Timer()
    timer.tic()

    args = parse_args()
    print('Called with args:')
    print(args)

    print('Using options:')
    pprint.pprint(options)

    caffe.set_device(args.gpu_id)
    caffe.set_mode_gpu()

    # train segmentation
    method = 'segment'
    cachepath = osp.join(options.cachepath, method)
    if not osp.isdir(cachepath):
        os.makedirs(cachepath)

    for dataset, trainset, num_classes in zip(options.seg.datasets,options.seg.trainsets, options.seg.num_classes):
        for net, pad in zip(options.seg.nets, options.seg.netpads):
            train_model(dataset, trainset, num_classes, net, pad, cachepath)
            if timer.toc(False) > options.timelimit:
                print 'Time limit!'
                sys.exit(1991)
    
