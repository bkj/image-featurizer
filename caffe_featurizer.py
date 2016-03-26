import os
import sys
import numpy as np
import caffe

class bcolors:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'


class CaffeFeaturizer:
    net         = None
    transformer = None
    files       = []
    batch_size  = None
    quiet       = None
    counter     = 0
    
    prototxt   = 'models/bvlc_reference_caffenet/deploy.prototxt'
    caffemodel = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    meanimage  = 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
    
    def __init__(self, CAFFE_ROOT, quiet = False):
        self.caffe_root = CAFFE_ROOT
        caffe.set_mode_cpu()
        self.net    = caffe.Net(self.caffe_root + self.prototxt, self.caffe_root + self.caffemodel, caffe.TEST)
        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        
        if self.meanimage:
            transformer.set_mean('data', np.load(self.caffe_root + self.meanimage).mean(1).mean(1))
        
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))
        self.transformer = transformer
        self.quiet = quiet
        
    def set_batch_size(self, n):
        self.batch_size = n
        self.net.blobs['data'].reshape(n, 3, 227, 227)

    def get_files(self):
        return self.files

    def set_files(self, files):
        self.files = files

    def load_files(self, print_mod = 50):
        self.errs = []
        i = 0
        for f in self.files:
            if (i + 1) % print_mod == 0:
                if not self.quiet:
                    print >> sys.stderr, bcolors.OKGREEN + 'total: %d \t current batch: %d' % (self.counter, i) + bcolors.ENDC
            
            try:
                self.net.blobs['data'].data[i] = self.transformer.preprocess('data', caffe.io.load_image(f))
            except:
                if not self.quiet:
                    print >> sys.stderr, bcolors.WARNINGS + 'error at %s (%d)' % (f, i) + bcolors.ENDC
                self.errs.append(i)
            
            i       += 1
            self.counter += 1

    def forward(self):
        if not self.quiet:
            print >> sys.stderr, bcolors.OKBLUE + ' -- forward pass -- ' + bcolors.ENDC + '\n'
        self.net.forward()

    def featurize(self, layer = 'fc7'):
        feat = [ self.net.blobs['fc7'].data[i] for i in range(self.batch_size) ]
        feat = np.array(feat)
        return feat

