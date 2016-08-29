'''
This script uses a pretrained PHOCNet to generate the
output for a given test list
'''
import argparse       
from phocnet.evaluation.phocnet_evaluator import PHOCNetEvaluation

if __name__ == '__main__':
    pne = PHOCNetEvaluation()
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()
    
    standard_args = argparse.ArgumentParser(add_help=False)
    standard_args.add_argument('--phocnet_bin_path', action='store', type=str, required=True,
                                   help='Path to a pretrained PHOCNet binary protofile.')    
    standard_args.add_argument('--doc_img_dir', action='store', type=str, required=True,
                                   help='The location of the document images.')
    standard_args.add_argument('--gpu_id', '-gpu', action='store', type=int, 
                                   help='The ID of the GPU to use. If not specified, prediction is run in CPU mode.')
    standard_args.add_argument('--debug_mode', '-dm', action='store_true', default=False,
                                   help='Flag indicating to run the PHOCNet training in debug mode.')
    standard_args.add_argument('--train_xml_file', action='store', type=str, required=True,
                               help='The READ-style XML file of words used in training')
    standard_args.add_argument('--test_xml_file', action='store', type=str, required=True,
                               help='The READ-style XML file of words to be used for testing')
    standard_args.add_argument('--deploy_proto_path', '-dpp', action='store', type=str, default='/tmp/deploy_phocnet.prototxt',
                               help='Location where to save the deploy file. Default: /tmp/deploy_phocnet.prototxt')
    standard_args.add_argument('--metric', action='store', type=str, default='braycurtis',
                               help='The metric to be used when comparing the PHOCNet output. Possible: all scipy metrics. Default: braycurtis')
    standard_args.add_argument('--phoc_unigram_levels', '-pul', action='store', 
                               type=lambda str_list: [int(elem) for elem in str_list.split(',')], 
                               default='2,3,4,5', 
                               help='The comma seperated list of PHOC unigram levels. Default: 2,3,4,5')
    standard_args.add_argument('--no_bigrams', '-ub', action='store_true', default=False,
                               help='Flag indicating to build the PHOC without bigrams')
    standard_args.add_argument('--annotation_delimiter', '-ad', action='store', type=str,
                               help='If the annotation in the XML is separated by special delimiters, it can be specified here.')
    
    # --- calc-phocs
    extract_unigrams_parser = subparser.add_parser('extract-unigrams', help='Finds the unigrams for a given XML and saves them')
    calc_phocs_parser = subparser.add_parser('calc-phocs', help='Predicts PHOCs from a pretrained CNN and saves them',
                                             parents=[standard_args])
    calc_phocs_parser.add_argument('--output_dir', '-od', action='store', type=str, default='.',
                                   help='Location where to save the estimated PHOCs. Default: working directory')
    calc_phocs_parser.set_defaults(func=pne.predict_and_save_phocs)
    
    # --- extract-unigrams
    extract_unigrams_parser.add_argument('--word_xml_file', action='store', type=str, required=True,
                                         help='The READ-style XML file of words for which to extract the unigrams')
    extract_unigrams_parser.add_argument('--annotation_delimiter', '-ad', action='store', type=str,
                                         help='If the annotation in the XML is separated by special delimiters, it can be specified here.')
    extract_unigrams_parser.add_argument('--doc_img_dir', action='store', type=str, required=True,
                                         help='The location of the document images.')  
    extract_unigrams_parser.set_defaults(func=pne.extract_unigrams)
    
    # --- qbs
    qbs_parser = subparser.add_parser('qbs', help='Evaluates the supplied CNN in a Query-by-String scenario (for protocol see Sudholt 2016 - PHOCNet)',
                                      parents=[standard_args])
    qbs_parser.set_defaults(func=pne.eval_qbs)
    
    # --- qbe
    qbe_parser = subparser.add_parser('qbe', help='Evaluates the supplied CNN in a Query-by-Example scenario (for protocol see Sudholt 2016 - PHOCNet)',
                                      parents=[standard_args])  
    qbe_parser.set_defaults(func=pne.eval_qbe)    
           
    # run everything   
    subfunc = parser.parse_args()
    args = vars(subfunc).copy()
    del args['func']
    subfunc.func(**args)