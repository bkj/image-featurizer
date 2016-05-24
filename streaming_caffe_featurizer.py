#!/usr/bin/python

# Streaming Caffe Image Featurizer
#
# Call like:
#
# ls images | xargs -I {} echo "images/{}" | python streaming_caffe_featurizer.py --sparse >> output.csv 
#

import os
import sys
import itertools
import numpy as np
import pandas as pd

CAFFE_ROOT = '/home/bjohnson/caffe/'
sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe

from caffe_featurizer import CaffeFeaturizer

prototxt   = CAFFE_ROOT + 'models/bvlc_reference_caffenet/deploy.prototxt'
caffemodel = CAFFE_ROOT + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
meanimage  = CAFFE_ROOT + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
cf = CaffeFeaturizer(prototxt, caffemodel, meanimage, mode='gpu')

# --
# I/O Functions
def chunker(stream, CHUNK_SIZE=100):
    out = []
    for x in stream:
        out.append(x.strip())
        if len(out) == CHUNK_SIZE:
            yield out
            out = []
    
    yield out

# NB : Sparse repeats file names a lot, which is bad
# Should print small keys and then a filename - key map
def writer(proc):
    mids = proc['id']
    del proc['id']
    out = proc.T.to_dict()
    for mid,v in zip(mids, out.itervalues()):
        print '%s\t%s' % (mid, ' '.join(['%s:%s' % (kk, vv) for kk,vv in v.iteritems() if vv != 0]))

# --

def process_chunk(chunk):
    cf.set_batch_size(len(chunk))        # Set batch size
    cf.set_files(itertools.chain(chunk)) # Give iterator for files
    cf.load_files()                      # Load the files from disk
    cf.forward()                         # Forward pass of NN
    
    feats = pd.DataFrame(cf.featurize()) # Dataframe of features
    cols = list(feats.columns)
    
    feats['id'] = chunk
    
    feats       = feats[['id'] + cols]   # Column of ids
    feats       = feats.drop(cf.errs)    # Error handling
    return feats

if __name__ == "__main__" :
    for chunk in chunker(sys.stdin):
        writer(process_chunk(chunk))

