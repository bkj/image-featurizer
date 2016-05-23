#!/usr/bin/python

# Streaming Caffe Image Featurizer
#
# Call like:
#
# ls images | xargs -I {} echo "images/{}" | python streaming_caffe_featurizer.py --sparse >> output.csv 


import os, sys, itertools
import numpy as np
import pandas as pd
import argparse

PROJECT_ROOT = '/Users/BenJohnson/projects/caffe_featurize'
CAFFE_ROOT   = '/Users/BenJohnson/projects/software/caffe/'
sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe

sys.path.append(PROJECT_ROOT)
from caffe_featurizer import CaffeFeaturizer
cf = CaffeFeaturizer(CAFFE_ROOT)

# --
# Argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sparse', dest = 'sparse', action="store_true")

args = parser.parse_args()

# --
# I/O Functions
def chunker(stream, CHUNK_SIZE = 250):
    out = []
    for x in stream:
        out.append(x.strip())
        if len(out) == CHUNK_SIZE:
            yield out
            out = []
    
    yield out

# NB : Sparse repeats file names a lot, which is bad
# Should print small keys and then a filename - key map
def writer(proc, sparse = False):
    if not sparse:
        print proc.to_csv(header = False, index = False)
    else:
        df = pd.melt(proc, 'id')
        df = df[df.value != 0].reset_index()
        del df['index']
        print df.to_csv(header = False, index = False)

# --

def process_chunk(chunk):
    cf.set_batch_size(len(chunk))        # Set batch size
    cf.set_files(itertools.chain(chunk)) # Give iterator for files
    cf.load_files()                      # Load the files from disk
    cf.forward()                         # Forward pass of NN
    
    feats       = pd.DataFrame(cf.featurize()) # Dataframe of features
    cols        = list(feats.columns)
    
    feats['id'] = chunk
    
    feats       = feats[['id'] + cols]   # Column of ids
    feats       = feats.drop(cf.errs)    # Error handling
    return feats

if __name__ == "__main__" :
    for chunk in chunker(sys.stdin):
        writer(process_chunk(chunk), args.sparse)

