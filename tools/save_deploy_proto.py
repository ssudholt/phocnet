#!/usr/bin/env python
import argparse
import os
from phocnet.caffe.model_proto_generator import ModelProtoGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save a PHOCNet deploy proto file to disk.')
    parser.add_argument('--output_dir', '-od', action='store', type=str, default='.',
                        help='The directory where to save the deploy proto. Default: .')
    parser.add_argument('--phoc_size', '-ps', action='store', type=int, default=604,
                        help='The dimensionality of the PHOC. Default: 604')
    args = vars(parser.parse_args())
    proto = ModelProtoGenerator(use_cudnn_engine=False).get_phocnet(word_image_lmdb_path=None, phoc_lmdb_path=None, 
                                                                    phoc_size=args['phoc_size'], generate_deploy=True)
    with open(os.path.join(args['output_dir'], 'deploy_phocnet.prototxt'), 'w') as deploy_file:
        deploy_file.write('#Deploy PHOCNet\n')
        deploy_file.write(str(proto))