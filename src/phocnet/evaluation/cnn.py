'''
Created on Jul 10, 2016

@author: ssudholt
'''
import logging

import numpy as np
from skimage.transform import resize

from phocnet.evaluation.retrieval import map_from_feature_matrix

def net_output_for_word_image_list(phocnet, word_img_list, 
                                   min_img_width_height=-1,input_layer='word_images', 
                                   output_layer='sigmoid', print_frequency=1000):
    '''
    Predict PHOCs from the given PHOCNet
    @param phocnet: caffe.Net
        A pretrained PHOCNet. The first layer of the PHOCNet must be an InputLayer
        (no LMDB or MemoryDataLayers)
    @param word_img_list: list of ndarrays
        A list of word images for which to predict the PHOCs.
        Every image in the last has to be a single channel gray-scale or binary
        ndarray in the range from 0 (black) to 255 (white).
    @param min_img_width_height: int
        The minimum height or width of an image to be passed to the PHOCNet.
        If an image in the word_img_list is smaller than the supplied number
        it is automatically resized before processed by the CNN. Default: -1
    @param input_layer: str
        The name of the input layer blob. Default: word_images 
    @param output_layer: str
        The name of the output layer blob. Default: sigmoid
    @param print_frequency: int
        Output is generated after this amount of images has been prcessed by
        the PHOCNet.
    '''
    output = []
    logger = logging.getLogger('NetOutput')
    logger.info('Evaluating net...')
    for idx, word_img in enumerate(word_img_list):
        # scale to correct pixel values (0 = background, 1 = text)
        word_img = word_img.astype(np.float32)
        word_img -= 255.0
        word_img /= -255.0      
              
        # check size
        if np.amin(word_img.shape[:2]) < min_img_width_height:
            scale = float(min_img_width_height+1)/float(np.amin(word_img.shape[:2]))
            new_shape = (int(scale*word_img.shape[0]), int(scale*word_img.shape[1]))
            word_img = resize(image=word_img, output_shape=new_shape)
        word_img = word_img.reshape((1,1,) + word_img.shape).astype(np.float32)            
        
        # reshape the PHOCNet
        phocnet.blobs[input_layer].reshape(*word_img.shape)
        phocnet.reshape()
        
        # forward the word image through the PHOCNet
        phocnet.blobs[input_layer].data[...] = word_img
        output.append(phocnet.forward()[output_layer].flatten())
        if ((idx+1)%print_frequency == 0 or (idx+1) == len(word_img_list)):
            logger.debug('    [ %*d / %d ]', len(str(len(word_img_list))), idx+1, len(word_img_list))            
    return np.vstack(output)

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