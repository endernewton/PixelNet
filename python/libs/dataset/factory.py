from dataset.segvoc_aug import segvoc_aug
from dataset.segpascalcontext import segpascalcontext, segpascalcontext33, segpascalcontext20

__sets = {}

for split in ['train', 'val']:
    name = 'context_{}'.format(split)
    __sets[name] = (lambda split=split:
        segpascalcontext(split))

for split in ['train', 'val']:
    name = 'context33_{}'.format(split)
    __sets[name] = (lambda split=split:
        segpascalcontext33(split))

for split in ['train', 'val']:
    name = 'context20_{}'.format(split)
    __sets[name] = (lambda split=split:
        segpascalcontext20(split))

for split in ['train', 'val', 'trainval', 'test']:
    name = 'aug-voc2012_{}'.format(split)
    __sets[name] = (lambda split=split:
        segvoc_aug(split))

def get_imdb(name):
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()