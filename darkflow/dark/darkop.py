from .layer import Layer
from .convolution import *
from .connected import *

class avgpool_layer(Layer):
    pass

class crop_layer(Layer):
    pass

class maxpool_layer(Layer):
    def setup(self, ksize, stride, pad, option):
        self.stride = stride
        self.ksize = ksize
        self.pad = pad

class softmax_layer(Layer):
    def setup(self, groups, option):
        self.groups = groups

class dropout_layer(Layer):
    def setup(self, p, option):
        self.h['pdrop'] = dict({
            'feed': p, # for training
            'dfault': 1.0, # for testing
            'shape': ()
        })

class route_layer(Layer):
    def setup(self, routes, option):
        self.routes = routes

class reorg_layer(Layer):
    def setup(self, stride, option):
        self.stride = stride

class parallel_merge_layer(Layer):
    pass

"""
Darkop Factory
"""

darkops = {
    'dropout': dropout_layer,
    'connected': connected_layer,
    'maxpool': maxpool_layer,
    'convolutional': convolutional_layer,
    'avgpool': avgpool_layer,
    'softmax': softmax_layer,
    'crop': crop_layer,
    'local': local_layer,
    'select': select_layer,
    'route': route_layer,
    'reorg': reorg_layer,
    'conv-select': conv_select_layer,
    'conv-extract': conv_extract_layer,
    'extract': extract_layer,
    'parallel-merge': parallel_merge_layer
}

def create_darkop(ltype, num, *args):
    op_class = darkops.get(ltype, Layer)
    return op_class(ltype, num, *args)