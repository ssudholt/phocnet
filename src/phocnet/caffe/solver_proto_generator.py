'''
Created on Jul 9, 2016

@author: ssudholt
'''
from caffe.proto import caffe_pb2
from google.protobuf.internal.containers import RepeatedScalarFieldContainer

def generate_solver_proto(**kwargs):
    sp = caffe_pb2.SolverParameter()
    for k,v in kwargs.iteritems():
        if not hasattr(sp, k):
            raise ValueError('The argument \'%s\' is not part of the Caffe solver parameters!')
        elif v is not None:
            elem = getattr(sp, k)
            if type(elem) == RepeatedScalarFieldContainer:
                elem.append(v)
            elif k == 'solver_mode':
                setattr(sp, k, sp.SolverMode.DESCRIPTOR.values_by_name[v].number)
            else:
                setattr(sp, k, v)
    return sp