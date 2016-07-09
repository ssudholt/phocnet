'''
Created on Aug 29, 2014

This module holds simple container classes for storing
word images for the PHOCNet experiments

@author: ssudholt
'''
import cv2
import numpy as np


class SimpleWordContainer(object):
    def __init__(self, transcription, bounding_box, image_path):
        self.__transcription = transcription
        self.__bounding_box = bounding_box
        self.__image_path = image_path

    def get_transcription(self):
        return self.__transcription

    def get_bounding_box(self):
        return self.__bounding_box


    def get_image_path(self):
        return self.__image_path


    def set_transcription(self, value):
        self.__transcription = value


    def set_bounding_box(self, value):
        self.__bounding_box = value


    def set_image_path(self, value):
        self.__image_path = value


    def del_transcription(self):
        del self.__transcription


    def del_bounding_box(self):
        del self.__bounding_box


    def del_image_path(self):
        del self.__image_path
    
    def get_word_image(self, gray_scale=True):
        col_type = None
        if gray_scale:
            col_type = cv2.CV_LOAD_IMAGE_GRAYSCALE
        else:
            col_type = cv2.CV_LOAD_IMAGE_COLOR
        
        # load the image
        ul = self.bounding_box['upperLeft']
        wh = self.bounding_box['widthHeight']
        img = cv2.imread(self.image_path, col_type)
        if not np.all(self.bounding_box['widthHeight'] == -1):
            img = img[ul[1]:ul[1]+wh[1], ul[0]:ul[0]+wh[0]]
        return img

    transcription = property(get_transcription, set_transcription, del_transcription, "transcription's docstring")
    bounding_box = property(get_bounding_box, set_bounding_box, del_bounding_box, "bounding_box's docstring")
    image_path = property(get_image_path, set_image_path, del_image_path, "image_path's docstring")

class DocImageWordContainer(SimpleWordContainer):
    def __init__(self, transcription, page, bounding_box,  
                 id_on_page, image_path):
        super(DocImageWordContainer, self).__init__(transcription, bounding_box, image_path)
        self.__page = page
        self.__id_on_page = id_on_page

    def get_page(self):
        return self.__page


    def get_id_on_page(self):
        return self.__id_on_page


    def set_page(self, value):
        self.__page = value


    def set_id_on_page(self, value):
        self.__id_on_page = value


    def del_page(self):
        del self.__page


    def del_id_on_page(self):
        del self.__id_on_page

    page = property(get_page, set_page, del_page, "page's docstring")
    id_on_page = property(get_id_on_page, set_id_on_page, del_id_on_page, "id_on_page's docstring")    
        