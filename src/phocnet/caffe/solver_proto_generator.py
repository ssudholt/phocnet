'''
Created on Jul 9, 2016

@author: ssudholt
'''
from caffe.proto import caffe_pb2

def generate_solver_proto(**kwargs):
    sp = caffe_pb2.SolverParameter()
    for k,v in kwargs.iteritems():
        if not hasattr(sp, k):
            raise ValueError('The argument \'%s\' is not part of the Caffe solver parameters!')
        elif v is not None:
            setattr(sp, k, v)
    return sp