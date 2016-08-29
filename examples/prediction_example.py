import caffe
import numpy as np

def main():
    # This example is going to show you how you can use the API to predict
    # PHOCs from a trained PHOCNet for your own word images.

    # First we need to load the trained PHOCNet. We are going to use the trained
    # PHOCNet supplied at 
    # http://patrec.cs.tu-dortmund.de/files/cnns/phocnet_gw_cv1.binaryproto
    deploy_path = 'deploy_phocnet.prototxt'
    trainet_net_path = 'phocnet_gw_cv1.binaryproto'
    phocnet = caffe.Net(deploy_path, caffe.TEST, weights=trainet_net_path)
    
    # Now you can supply your own images. For the sake of example, we use
    # random arrays. We generate 4 images of shape 60 x 160, each having one
    # channel. The pixel range is 0 - 255
    images = [np.around(np.random.rand(60, 160, 1)*255)
              for _ in xrange(4)]
    
    # Note that the image ndarray arrays are now in the typical shape and pixel
    # range of what you would get if you were to load your images with the
    # standard tools such as OpenCV or skimage. For Caffe, we need to translate
    # it into a 4D tensor of shape (num. images, channels, height, width)
    for idx in xrange(4):
        images[idx] = np.transpose(images[idx], (2, 0, 1))
        images[idx] = np.reshape(images[idx], (1, 1, 60, 160))
    
        # The PHOCNet accepts images in a pixel range of 0 (white) to 1 (black).
        # Typically, the pixel intensities are inverted i.e. white is 255 and
        # black is 0. We thus need to prepare our word images to be in the
        # correct range. If your images are already in 0 (white) to 1 (black)
        # you can skip this step.
        images[idx] -= 255.0
        images[idx] /= -255.0
    
    # Now we are all set to shove the images through the PHOCNet.
    # As we usually have different image sizes, we need to predict them
    # one by one from the net.
    # First, you need to reshape the input layer blob (word_images) to match 
    # the current word image shape you want to process.
    phocs = []
    for image in images:
        phocnet.blobs['word_images'].reshape(*image.shape)
        phocnet.reshape()
    
        # Put the current image into the input layer...
        phocnet.blobs['word_images'].data[...] = image
    
        # ... and predict the PHOC (flatten automatically returns a copy)
        phoc = phocnet.forward()['sigmoid'].flatten()
        phocs.append(phoc)
    
    # Congrats, you have a set of PHOCs for your word images.
    # If you run into errors with the code above, make sure that your word images are
    # shape (num. images, channels, height, width).
    # Only in cases where you have images of the exact same size should num. images
    # be different from 1    
    
if __name__ == '__main__':
    main()