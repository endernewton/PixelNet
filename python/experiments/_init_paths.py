import os.path as osp
import sys
import commands

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# check which caffe folder to include
res = commands.getstatusoutput('nvidia-smi')
# CPU
if res[0]:
    caffe_path = osp.abspath(osp.join(this_dir, '..', '..', 'cpu48', 'python'))
# GPU
else:
    caffe_path = osp.abspath(osp.join(this_dir, '..', '..', 'caffe48', 'python'))

add_path(caffe_path)

# then add the library
lib_path = osp.join(this_dir, '..', 'libs')
add_path(lib_path)

