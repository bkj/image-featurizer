#!/usr/bin/python

import os, sys, itertools
import numpy as np
import pandas as pd
import argparse

CAFFE_ROOT = '/Users/BenJohnson/projects/software/caffe/'
sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe

sys.path.append('/Users/BenJohnson/projects/caffe_featurize')
from caffe_featurizer import CaffeFeaturizer
cf = CaffeFeaturizer(CAFFE_ROOT)

# --
# Argparse

parser = argparse.ArgumentParser()
parser.add_argument('--print-pandas', dest = 'print_pandas', action="store_true")

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

def writer(proc, print_pandas = False):
    if print_pandas:
        print proc
    else:
        print proc.to_csv(header = False)

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
        writer(process_chunk(chunk), args.print_pandas)