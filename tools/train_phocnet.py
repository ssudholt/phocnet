import argparse
from phocnet.training.phocnet_trainer import PHOCNetTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required training parameters
    parser.add_argument('--doc_img_dir', action='store', type=str, required=True,
                      help='The location of the document images.')
    parser.add_argument('--train_annotation_file', action='store', type=str, required=True,
                      help='The file path to the READ-style XML file for the training partition of the dataset to be used.')
    parser.add_argument('--test_annotation_file', action='store', type=str, required=True,
                      help='The file path to the READ-style XML file for the testing partition of the dataset to be used.')
    parser.add_argument('--proto_dir', action='store', type=str, required=True,
                      help='Directory where to save the protobuffer files generated during the training.')
    parser.add_argument('--lmdb_dir', action='store', type=str, required=True,
                      help='Directory where to save the LMDB databases created during training.')
    # IO parameters
    parser.add_argument('--save_net_dir', '-snd', action='store', type=str,
                      help='Directory where to save the final PHOCNet. If unspecified, the net is not saved after training')
    parser.add_argument('--recreate_lmdbs', '-rl', action='store_true', default=False,
                      help='Flag indicating to delete existing LMDBs for this dataset and recompute them.')
    parser.add_argument('--debug_mode', '-dm', action='store_true', default=False,
                      help='Flag indicating to run the PHOCNet training in debug mode.')
    # Caffe parameters
    parser.add_argument('--learning_rate', '-lr', action='store', type=float, default=0.0001, 
                      help='The learning rate for SGD training. Default: 0.0001')
    parser.add_argument('--momentum', '-mom', action='store', type=float, default=0.9,
                      help='The momentum for SGD training. Default: 0.9')
    parser.add_argument('--step_size', '-ss', action='store', type=int, default=70000, 
                      help='The step size at which to reduce the learning rate. Default: 70000')
    parser.add_argument('--display', action='store', type=int, default=500, 
                      help='The number of iterations after which to display the loss values. Default: 500')
    parser.add_argument('--test_interval', action='store', type=int, default=500, 
                      help='The number of iterations after which to periodically evaluate the PHOCNet. Default: 500')
    parser.add_argument('--max_iter', action='store', type=int, default=80000, 
                      help='The maximum number of SGD iterations. Default: 80000')
    parser.add_argument('--batch_size', '-bs', action='store', type=int, default=10, 
                      help='The batch size after which the gradient is computed. Default: 10')
    parser.add_argument('--weight_decay', '-wd', action='store', type=float, default=0.00005,
                      help='The weight decay for SGD training. Default: 0.00005')
    parser.add_argument('--gamma', '-gam', action='store', type=float, default=0.1,
                       help='The value with which the learning rate is multiplied after step_size iteraionts. Default: 0.1')
    parser.add_argument('--gpu_id', '-gpu', action='store', type=int, 
                      help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
    # PHOCNet parameters
    parser.add_argument('--phoc_unigram_levels', '-pul', action='store', type=lambda x: [int(elem) for elem in x.split(',')], default='2,3,4,5',
                      help='Comma seperated list of PHOC unigram levels to be used. Default: 2,3,4,5')
    parser.add_argument('--use_bigrams', '-ub', action='store_true',
                        help='Flag indicating to build the PHOC with bigrams')
    parser.add_argument('--n_train_images', '-nti', action='store', type=int, default=500000, 
                      help='The number of images to be generated for the training LMDB. Default: 500000')
    parser.add_argument('--metric', '-met', action='store', type=str, default='braycurtis',
                      help='The metric with which to compare the PHOCNet predicitions (possible metrics are all scipy metrics). Default: braycurtis')
    parser.add_argument('--annotation_delimiter', '-ad', action='store', type=str,
                      help='If the annotation in the XML is separated by special delimiters, it can be specified here.')
    parser.add_argument('--use_lower_case_only', '-ulco', action='store_true', default=False,
                      help='Flag indicating to convert all annotations from the XML to lower case before proceeding')
    
    params = vars(parser.parse_args())
    PHOCNetTrainer(**params).train_phocnet()