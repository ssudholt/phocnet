'''
Created on Feb 18, 2016

@author: ssudholt
'''
import numpy as np
import cv2

class AugmentationCreator(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        pass
    
    @staticmethod
    def create_affine_transform_augmentation(img, random_limits=(0.8, 1.1)):
        '''
        Creates an augmentation by computing a homography from three
        points in the image to three randomly generated points
        '''
        y, x = img.shape[:2]
        fx = float(x)
        fy = float(y)
        src_point = np.float32([[fx/2, fy/3,],
                                [2*fx/3, 2*fy/3],
                                [fx/3, 2*fy/3]])
        random_shift = (np.random.rand(3,2) - 0.5) * 2 * (random_limits[1]-random_limits[0])/2 + np.mean(random_limits)
        dst_point = src_point * random_shift.astype(np.float32)
        transform = cv2.getAffineTransform(src_point, dst_point)
        borderValue = 0
        if img.ndim == 3:
            borderValue = np.median(np.reshape(img, (img.shape[0]*img.shape[1],-1)), axis=0)
        else:
            borderValue=np.median(img)
        warped_img = cv2.warpAffine(img, transform, dsize=(x,y), borderValue=borderValue)
        return warped_img
    
    @staticmethod
    def create_random_noise_flip_shift_rotate_augmentation(img, noise_variance=0.02, max_shift=0.05,
                                                           max_abs_rotation=10, obj_list=None, obj_type = "rectangle"):
        '''
        Creates an augmentation by randomly flipping the image,
        applying random noise from a Gaussian distribution, shifting the image
        and rotating it.
        '''
        # copy the image
        aug_img = img.copy()
        
        # gaussian noise
        aug_img = aug_img.astype(np.float32)
        aug_img += (np.random.normal(loc=0.0, scale=noise_variance, size=img.shape)*255)        
        # bring back to correct range
        aug_img -= aug_img.min()
        aug_img *= 255.0/aug_img.max()
        aug_img = aug_img.astype(np.uint8)
        
        # flip
        if np.random.rand() > 0.5:
            aug_img = AugmentationCreator.flip_image_lr(aug_img)
            
            if obj_list != None:
                if obj_type == "rectangle":
                    obj_list = AugmentationCreator.flip_bboxes_lr(obj_list)
                elif obj_type == "point":
                    obj_list = AugmentationCreator.flip_points_lr(obj_list)
        
        # random rotation
        angle = int((np.random.rand() - 0.5) * max_abs_rotation)
        aug_img = AugmentationCreator.rotate_image(aug_img, angle)
        
        if obj_list != None:
            if obj_type == "rectangle":
                obj_list = AugmentationCreator.rotate_bboxes(img, obj_list, angle)
            elif obj_type == "point":
                obj_list = AugmentationCreator.rotate_points(img, obj_list, angle)
        
        # random translation
        translation = (np.random.rand(2)-0.5) * max_shift
        aug_img = AugmentationCreator.translate_image(aug_img, translation)
        
        if obj_list != None:
            if obj_type == "rectangle":
                obj_list = AugmentationCreator.translate_bboxes(obj_list, translation)
            elif obj_type == "point":
                obj_list = AugmentationCreator.translate_points(obj_list, translation)
        
        return aug_img, obj_list, angle
    
    @staticmethod
    def flip_image_lr(image):
        '''
        Flips the given image vertically.
        '''
        
        return np.fliplr(image)
    
    @staticmethod
    def flip_points_lr(obj_list):
        '''
        Flips the points of the given objects vertically. The coordinates of the points have to be
        normalized.
        '''
        
        flipped_obj_list = []
        for obj in obj_list:
            obj_name = obj[0]
            x = obj[1][0]
            y = obj[1][1]
            flipped_obj_list.append((obj_name, (1 - x, y)))
        
        return flipped_obj_list
    
    @staticmethod
    def flip_bboxes_lr(obj_list):
        '''
        Flips the bounding boxes of the given objects vertically. The coordinates of the bounding
        boxes have to be normalized.
        '''
        
        flipped_obj_list = []
        for obj in obj_list:
            obj_name = obj[0]
            upper_left = obj[1]['upper_left']
            lower_right = obj[1]['lower_right']
            upper_left = np.array([1 - upper_left[0], upper_left[1]])
            lower_right = np.array([1 - lower_right[0], lower_right[1]])
            flipped_obj_list.append((obj_name, {'upper_left' : upper_left, 'lower_right' : lower_right}))
        
        return flipped_obj_list
    
    @staticmethod
    def rotate_image(image, angle):
        '''
        Rotates the given image by the given angle.
        '''
        
        rows, cols, _ = np.atleast_3d(image).shape
        rot_mat = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        
        return cv2.warpAffine(image, rot_mat, (cols, rows))
    
    @staticmethod
    def rotate_points(image, obj_list, angle):
        '''
        Rotates the points of the given objects by the given angle. The points will be translated
        into absolute coordinates. Therefore the image (resp. its shape) is needed.
        '''
        
        rotated_obj_list = []
        cosOfAngle = np.cos(2 * np.pi / 360 * -angle)
        sinOfAngle = np.sin(2 * np.pi / 360 * -angle)
        image_shape = np.array(np.atleast_3d(image).shape[0:2][::-1])
        rot_mat = np.array([[cosOfAngle, -sinOfAngle], [sinOfAngle, cosOfAngle]])
        for obj in obj_list:
            obj_name = obj[0]
            point = obj[1] * image_shape
            rotated_point = AugmentationCreator._rotate_vector_around_point(image_shape/2, point, rot_mat) / image_shape
            rotated_obj_list.append((obj_name, (rotated_point[0], rotated_point[1])))
            
        return rotated_obj_list
    
    @staticmethod
    def rotate_bboxes(image, obj_list, angle):
        '''
        Rotates the bounding boxes of the given objects by the given angle. The bounding box will be
        translated into absolute coordinates. Therefore the image (resp. its shape) is needed.
        '''
        
        rotated_obj_list = []
        cosOfAngle = np.cos(2 * np.pi / 360 * -angle)
        sinOfAngle = np.sin(2 * np.pi / 360 * -angle)
        image_shape = np.array(np.atleast_3d(image).shape[0:2][::-1])
        rot_mat = np.array([[cosOfAngle, -sinOfAngle], [sinOfAngle, cosOfAngle]])
        for obj in obj_list:
            obj_name = obj[0]
            upper_left = obj[1]['upper_left'] * image_shape
            lower_right = obj[1]['lower_right'] * image_shape
            upper_left = AugmentationCreator._rotate_vector_around_point(image_shape/2, upper_left, rot_mat) / image_shape
            lower_right = AugmentationCreator._rotate_vector_around_point(image_shape/2, lower_right, rot_mat) / image_shape
            rotated_obj_list.append((obj_name, {'upper_left' : upper_left, 'lower_right' : lower_right}))
            
        return rotated_obj_list
    
    @staticmethod
    def _rotate_vector_around_point(point, vector, rot_mat):
        '''
        Rotates a given vector around the given point using the given rotation matrix.
        '''
        
        centering_translation = np.array([point[0], point[1]])
        rotated_vector = vector - centering_translation
        rotated_vector = np.dot(rot_mat, rotated_vector.reshape(2, 1)).reshape(2)
        rotated_vector += centering_translation
        
        return rotated_vector
    
    @staticmethod
    def translate_image(image, translation):
        '''
        Translates the given image with the given translation vector.
        '''
        
        rows, cols, _ = np.atleast_3d(image).shape
        trans_mat = np.array([[1, 0, translation[0]*cols], [0, 1, translation[1]*rows]])
        
        return cv2.warpAffine(image, trans_mat, (cols, rows))
    
    @staticmethod
    def translate_points(obj_list, translation):
        '''
        Translates the points of the given objects with the given translation vector.
        '''
        
        translated_obj_list = []
        for obj in obj_list:
            obj_name = obj[0]
            point = obj[1]
            translated_point = point + translation
            translated_obj_list.append((obj_name, (translated_point[0], translated_point[1])))
        
        return translated_obj_list
    
    @staticmethod
    def translate_bboxes(obj_list, translation):
        '''
        Translates the bounding boxes of the given objects with the given translation vector.
        '''
        
        translated_obj_list = []
        for obj in obj_list:
            obj_name = obj[0]
            upper_left = obj[1]['upper_left']
            lower_right = obj[1]['lower_right']
            upper_left = upper_left + translation
            lower_right = lower_right + translation
            translated_obj_list.append((obj_name, {'upper_left' : upper_left, 'lower_right' : lower_right}))
        
        return translated_obj_list
