'''
Created on Jul 10, 2016

@author: ssudholt
'''
import numpy as np
from phocnet.evaluation.retrieval import map_from_feature_matrix

def calc_map_from_cnn_features(solver, test_iterations, metric):    
    net_output = np.zeros((test_iterations, solver.test_nets[0].blobs['sigmoid'].data.flatten().shape[0]))
    labels = np.zeros(test_iterations)
    for idx in xrange(solver.param.test_iter[0]):
            # calculate the net output
            solver.test_nets[0].forward()
            
            net_output[idx] = solver.test_nets[0].blobs['sigmoid'].data.flatten()
            labels[idx] = solver.test_nets[0].blobs['label'].data.flatten()    
    # calculate mAP
    _, ave_precs = map_from_feature_matrix(features=net_output, labels=labels, 
                                           metric=metric, drop_first=True)
    # some queries might not have a relevant sample in the test set
    # -> exclude them
    mean_ap = np.mean(ave_precs[ave_precs > 0])
    return mean_ap, ave_precs