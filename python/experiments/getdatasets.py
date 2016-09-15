import _init_paths
# from config.options import options
from dataset.factory import get_imdb

for split in ['train', 'val', 'trainval', 'test']:
    name = 'aug-voc2012_{}'.format(split)
    print 'DS: '+name
    imdb = get_imdb(name)

for split in ['train', 'val']:
    name = 'context_{}'.format(split)
    print 'DS: '+name
    imdb = get_imdb(name)

    name = 'context33_{}'.format(split)
    print 'DS: '+name
    imdb = get_imdb(name)

    name = 'context20_{}'.format(split)
    print 'DS: '+name
    imdb = get_imdb(name)