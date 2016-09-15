import _init_paths
from config.options import options
from dataset.factory import get_imdb
import pdb

for split in ['train', 'trainval']:
    name = 'aug-voc2012_{}'.format(split)
    print 'DS: '+name
    imdb = get_imdb(name)
    imdb.compute_stats()

for split in ['train']:
    name = 'context_{}'.format(split)
    print 'DS: '+name
    imdb = get_imdb(name)
    imdb.compute_stats()

    name = 'context33_{}'.format(split)
    print 'DS: '+name
    imdb = get_imdb(name)
    imdb.compute_stats()

    name = 'context20_{}'.format(split)
    print 'DS: '+name
    imdb = get_imdb(name)
    imdb.compute_stats()