#!/usr/bin/env python
'''
Script for predicting PHOCs for a number of images residing in a folder on disk.
'''
import argparse
import logging
import os

import caffe
import numpy as np
import cv2

from phocnet.evaluation.cnn import net_output_for_word_image_list

def main(img_dir, output_dir, pretrained_phocnet, deploy_proto, min_image_width_height, gpu_id):
	logging_format = '[%(asctime)-19s, %(name)s, %(levelname)s] %(message)s'
	logging.basicConfig(level=logging.INFO,
                        format=logging_format)
	logger = logging.getLogger('Predict PHOCs')
	
	if gpu_id is None:
		caffe.set_mode_cpu()
	else:
		caffe.set_mode_gpu()
		caffe.set_device(gpu_id)
	
	logger.info('Loading PHOCNet...')
	phocnet = caffe.Net(deploy_proto, caffe.TEST, weights=pretrained_phocnet)
	
	# find all images in the supplied dir
	logger.info('Found %d word images to process', len(os.listdir(img_dir)))
	word_img_list = [cv2.imread(os.path.join(img_dir, filename), cv2.CV_LOAD_IMAGE_GRAYSCALE) 
					 for filename in sorted(os.listdir(img_dir)) if filename not in ['.', '..']]
	# push images through the PHOCNet
	logger.info('Predicting PHOCs...')
	predicted_phocs = net_output_for_word_image_list(phocnet=phocnet, word_img_list=word_img_list, 
													min_img_width_height=min_image_width_height)
	# save everything
	logger.info('Saving...')
	np.save(os.path.join(output_dir, 'predicted_phocs.npy'), predicted_phocs)
	logger.info('Finished')
	
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Predict PHOCs from a pretrained PHOCNet. The PHOCs are saved as Numpy Array to disk.')
	parser.add_argument('--min_image_width_height', '-miwh', action='store', type=int, default=26,
						help='The minimum image width or height to be passed through the PHOCNet. Default: 26')
	parser.add_argument('--output_dir', '-od', action='store', type=str, default='.',
						help='The directory where to store the PHOC Numpy Array. Default: .')
	parser.add_argument('--img_dir', '-id', action='store', type=str, required=True,
						help='All images in this folder are processed in ASCII order of their '+
							 'respective names. A PHOC is predicted for each.')
	parser.add_argument('--pretrained_phocnet', '-pp', action='store', type=str, required=True,
						help='Path to a pretrained PHOCNet binaryproto file.')
	parser.add_argument('--deploy_proto', '-dp', action='store', type=str, required=True,
						help='Path to PHOCNet deploy prototxt file.')
	parser.add_argument('--gpu_id', '-gpu', action='store', type=int, 
						help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
	args = vars(parser.parse_args())
	main(**args)
	