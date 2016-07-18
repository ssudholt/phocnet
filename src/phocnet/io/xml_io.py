'''
Created on Jul 9, 2016

@author: ssudholt
'''
import logging
import os
import xml.etree.ElementTree as ET

import numpy as np

from phocnet.io.word_container import DocImageWordContainer

class XMLReader(object):
    '''
    Class for reading a training and test set from two READ-style XML files.
    '''
    def __init__(self, make_lower_case=True):
        '''
        Constructor
        
        Args:
            make_lower_case (bool): whether to convert XML annotation values to
                lower case
        '''
        self.logger = logging.getLogger('XMLReader')
        self.make_lower_case = make_lower_case
    
    def load_train_test_xml(self, train_xml_path, test_xml_path, img_dir):
        '''
        Reads two XML files as training and test partitions into word lists
        and returns them as well as the corresponding dataset name.
        
        Args:
            train_xml_path (str): the path to the READ-style training XML file
            test_xml_path (str): the path to the READ-style test XML file
        '''
        self.logger.info('Loading training XML at %s', train_xml_path)
        train_dataset_name, train_list = self.load_word_list_from_READ_xml(xml_filename=train_xml_path, img_dir=img_dir)
        
        self.logger.info('Loading test XML at %s', test_xml_path)
        test_dataset_name, test_list = self.load_word_list_from_READ_xml(xml_filename=test_xml_path, img_dir=img_dir)
        
        # check if the two datasets match
        if train_dataset_name != test_dataset_name:
            raise ValueError('Training and test XML do not belong to the same dataset!')
        return train_dataset_name, train_list, test_list
    

    def load_word_list_from_READ_xml(self, xml_filename, img_dir):
        self.logger.debug('Using XML-File at %s and image directory %s...', xml_filename, img_dir)
        tree = ET.parse(os.path.join(xml_filename))
        root = tree.getroot()
        # load the dataset name
        dataset_name = root.attrib['dataset']        
        # iterate through all word bounding boxes and put them in a word list
        word_list = []
        for word_idx, word_elem in enumerate(root.findall('spot')):
            transcription = unicode(word_elem.attrib['word'])
            if self.make_lower_case:
                transcription = transcription.lower()
            word_list.append(DocImageWordContainer(transcription=transcription, 
                                                   page=word_elem.attrib['image'].split('.')[0], 
                                                   bounding_box=dict(upperLeft=np.array([int(word_elem.attrib['x']), 
                                                                                         int(word_elem.attrib['y'])]),
                                                                     widthHeight=np.array([int(word_elem.attrib['w']), 
                                                                                           int(word_elem.attrib['h'])])), 
                                                   id_on_page=word_idx, 
                                                   image_path=os.path.join(img_dir, word_elem.attrib['image'])))
        return dataset_name, word_list                  