'''
Created on Jul 8, 2016

@author: ssudholt
'''
import os
from xml.etree import ElementTree

import numpy as np

from phocnet.io.word_container import DocImageWordContainer

def load_word_list_from_READ_xml(xml_file_path, doc_img_dir, dataset_name):
    tree = ElementTree.parse(os.path.join(xml_file_path))
    root = tree.getroot()
    # check if we have the correct XML
    if root.attrib['dataset'] != dataset_name:
        raise ValueError('The supplied XML file at \'%s\' is not for the %s dataset' % (xml_file_path, dataset_name))
    
    # iterate through all word bounding boxes and put them in a word list
    word_list = []
    for word_idx, word_elem in enumerate(root.findall('spot')):
        word_list.append(DocImageWordContainer(transcription=unicode(word_elem.attrib['word']).lower(), 
                                               page=word_elem.attrib['image'].split('.')[0], 
                                               bounding_box=dict(upperLeft=np.array([int(word_elem.attrib['x']), 
                                                                                     int(word_elem.attrib['y'])]),
                                                                 widthHeight=np.array([int(word_elem.attrib['w']), 
                                                                                       int(word_elem.attrib['h'])])), 
                                               id_on_page=word_idx, 
                                               image_path=os.path.join(doc_img_dir, word_elem.attrib['image'])))
    return word_list  