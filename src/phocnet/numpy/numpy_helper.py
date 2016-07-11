'''
Created on Feb 25, 2015

@author: ssudholt
'''
import numpy as np

class NumpyHelper(object):
        
    @staticmethod
    def get_unique_rows(arr, return_indices=False):
        '''
        Returns the unique rows of the supplied array
        this code was originally proposed at stackoverflow
        http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
        
        Args:
            arr (2d-ndarray): the array from which to extract the unique rows
            return_indices (bool): if true, the indices corresponding to the unique rows in arr are
                                   returned as well                        
        '''
        b = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
        _, idx = np.unique(b, return_index=True)
        
        # return the result
        if return_indices:
            return arr[idx], idx
        else:
            return arr[idx]
        