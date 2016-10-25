'''
Created on Jul 8, 2016

@author: ssudholt
'''
import logging

from caffe import NetSpec
from caffe import layers as L
from caffe import params as P
from caffe.io import caffe_pb2

class ModelProtoGenerator(object):
    '''
    Class for generating Caffe CNN models through protobuffer files.
    '''
    def __init__(self, initialization='msra', use_cudnn_engine=False):
        # set up the engines
        self.conv_engine = None
        self.spp_engine = None
        if use_cudnn_engine:
            self.conv_engine = P.Convolution.CUDNN
            self.spp_engine = P.SPP.CUDNN    
        else:
            self.conv_engine = P.Convolution.CAFFE
            self.spp_engine = P.SPP.CAFFE
        self.phase_train = caffe_pb2.Phase.DESCRIPTOR.values_by_name['TRAIN'].number
        self.phase_test = caffe_pb2.Phase.DESCRIPTOR.values_by_name['TEST'].number
        self.initialization = initialization
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def conv_relu(self, bottom, nout, kernel_size=3, stride=1, pad=1, relu_in_place=True):
        '''
        Helper method for returning a ReLU activated Conv layer
        '''    
        conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride,
                             num_output=nout, pad=pad, engine=self.conv_engine, 
                             weight_filler=dict(type=self.initialization), 
                             bias_filler=dict(type='constant'))
        return conv, L.ReLU(conv, in_place=relu_in_place)
    
    def fc_relu(self, bottom, layer_size, dropout_ratio=0.0, relu_in_place=True):
        '''
        Helper method for returning a ReLU activated Fully Connected layer. It can be specified also
        if the layer should make use of Dropout as well.
        '''
        fc = L.InnerProduct(bottom, num_output=layer_size,
                            weight_filler=dict(type=self.initialization), 
                            bias_filler=dict(type='constant'))
        relu = L.ReLU(fc, in_place=relu_in_place)
        if dropout_ratio == 0.0:
            return fc, relu
        else:
            return fc, relu, L.Dropout(relu, dropout_ratio=0.5, in_place=True, include=dict(phase=self.phase_train))
    
    def get_phocnet(self, word_image_lmdb_path, phoc_lmdb_path,                    
                    phoc_size=604, generate_deploy=False):
        '''
        Returns a NetSpec definition of the PHOCNet. The definition can then be transformed
        into a protobuffer message by casting it into a str.
        '''
        n = NetSpec()
        relu_in_place = True
        # Data
        if generate_deploy:
            n.word_images = L.Input(shape=dict(dim=[1,1,100,250]))
            relu_in_place = False
        else:
            n.word_images, n.label = L.Data(batch_size=1, backend=P.Data.LMDB, source=word_image_lmdb_path, prefetch=20,
                                            transform_param=dict(mean_value=255, scale=-1./255,), ntop=2)
            n.phocs, n.label_phocs = L.Data(batch_size=1, backend=P.Data.LMDB, source=phoc_lmdb_path, prefetch=20, 
                                            ntop=2)
        # Conv Part
        n.conv1_1, n.relu1_1 = self.conv_relu(n.word_images, nout=64, relu_in_place=relu_in_place)
        n.conv1_2, n.relu1_2 = self.conv_relu(n.relu1_1, nout=64, relu_in_place=relu_in_place)
        n.pool1 = L.Pooling(n.relu1_2, pooling_param=dict(pool=P.Pooling.MAX, kernel_size=2, stride=2))
        
        n.conv2_1, n.relu2_1 = self.conv_relu(n.pool1, nout=128, relu_in_place=relu_in_place)
        n.conv2_2, n.relu2_2 = self.conv_relu(n.relu2_1, nout=128, relu_in_place=relu_in_place)
        n.pool2 = L.Pooling(n.relu2_2, pooling_param=dict(pool=P.Pooling.MAX, kernel_size=2, stride=2))
        
        n.conv3_1, n.relu3_1 = self.conv_relu(n.pool2, nout=256, relu_in_place=relu_in_place)
        n.conv3_2, n.relu3_2 = self.conv_relu(n.relu3_1, nout=256, relu_in_place=relu_in_place)
        n.conv3_3, n.relu3_3 = self.conv_relu(n.relu3_2, nout=256, relu_in_place=relu_in_place)
        n.conv3_4, n.relu3_4 = self.conv_relu(n.relu3_3, nout=256, relu_in_place=relu_in_place)
        n.conv3_5, n.relu3_5 = self.conv_relu(n.relu3_4, nout=256, relu_in_place=relu_in_place)
        n.conv3_6, n.relu3_6 = self.conv_relu(n.relu3_5, nout=256, relu_in_place=relu_in_place)
        
        n.conv4_1, n.relu4_1 = self.conv_relu(n.relu3_6, nout=512, relu_in_place=relu_in_place)
        n.conv4_2, n.relu4_2 = self.conv_relu(n.relu4_1, nout=512, relu_in_place=relu_in_place)
        n.conv4_3, n.relu4_3 = self.conv_relu(n.relu4_2, nout=512, relu_in_place=relu_in_place)
        
        # FC Part
        n.spp5 = L.SPP(n.relu4_3, spp_param=dict(pool=P.SPP.MAX, pyramid_height=3, engine=self.spp_engine))
        n.fc6, n.relu6, n.drop6 = self.fc_relu(bottom=n.spp5, layer_size=4096, dropout_ratio=0.5, relu_in_place=relu_in_place)
        n.fc7, n.relu7, n.drop7 = self.fc_relu(bottom=n.drop6, layer_size=4096, dropout_ratio=0.5,relu_in_place=relu_in_place)        
        n.fc8 = L.InnerProduct(n.drop7, num_output=phoc_size, weight_filler=dict(type=self.initialization), bias_filler=dict(type='constant'))
        n.sigmoid = L.Sigmoid(n.fc8, include=dict(phase=self.phase_test))
        
        # output part
        if not generate_deploy:
            n.silence = L.Silence(n.sigmoid, ntop=0, include=dict(phase=self.phase_test))
            n.loss = L.SigmoidCrossEntropyLoss(n.fc8, n.phocs)            
            
        return n.to_proto()
        