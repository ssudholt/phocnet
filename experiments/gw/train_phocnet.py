'''
Created on Jul 8, 2016

@author: ssudholt
'''

class GWTrainPHOCNet(object):
    '''
    Experiment driver class for training the PHOCNet on the
    George Washington dataset.
    '''

    def __init__(self, **params):
        self.dataset_name = 'gw'
        super(GWTrainPHOCNet, self).__init__(train_annotation_file='',
                                             test_annotation_file='',
                                             params)
    
    def train_phocnet(self):
        pass
        
if __name__ == '__main__':
    
        