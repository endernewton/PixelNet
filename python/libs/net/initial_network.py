from config.options import options
import os.path as osp
import os

def initial_network(net):
    weightfolder = '/home/xinleic/ConT/fast-rcnn/data/imagenet_models/'
    modelfolder = '/home/xinleic/ConT/fast-rcnn/models/'

    if net.find('caffenet') >= 0:
        weightfile = osp.join(weightfolder,'CaffeNet.v2.caffemodel')
        modelfile = osp.join(modelfolder,'CaffeNet/test-o.prototxt')
    elif net.find('middle') >= 0:
        weightfile = osp.join(weightfolder,'VGG_CNN_M_1024.v2.caffemodel')
        modelfile = osp.join(modelfolder,'VGG_CNN_M_1024/test-o.prototxt')
    elif net.find('vgg16') >= 0:
        weightfile = osp.join(weightfolder,'VGG16.v2.caffemodel')
        modelfile = osp.join(modelfolder,'VGG16/test-o.prototxt')
    else:
        ex = NotImplementedError('Sorry, network not recognized!')
        raise ex

    savefolder = osp.join(options.cnetpath,net)
    if not osp.isdir(savefolder):
        osp.makedirs(savefolder)

    initmodel = osp.join(savefolder,net+'_init.caffemodel')
    if not osp.exists(initmodel):
        os.symlink(weightfile, initmodel)
    
    return weightfile, modelfile