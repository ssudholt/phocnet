import os
import shutil
import logging
import argparse
from subprocess import call
import sys

def main(cudnn_dir, no_caffe, opencv_dir, install_dir, install_caffe_dir):
    # init logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('install.py')
    
    # init submodules
    call(['git', 'submodule', 'init'])
    call(['git', 'submodule', 'update'])
    
    # compile caffe
    # cmake
    if not no_caffe:
        logger.info('Running CMake to configure Caffe submodule...')
        if install_caffe_dir is None:
            install_caffe_dir = os.path.join(install_dir, 'caffe')
        else: 
            install_caffe_dir = os.path.join(install_caffe_dir, 'caffe')
            
        os.chdir('caffe')
        if os.path.exists('build'):
            shutil.rmtree('build')
        os.makedirs('build')
        os.chdir('build')
        call_list = ['cmake', '..', '-DCMAKE_INSTALL_PREFIX=%s' % install_caffe_dir]
        if cudnn_dir is not None:
            call_list.append('-DCUDNN_DIR=%s' % cudnn_dir)
        if opencv_dir is not None:
            call_list.append('-DOpenCV_DIR=%s' % opencv_dir)
        if call(call_list) != 0:
            raise ValueError('Error during CMake run')
        # make
        logger.info('Compiling Caffe submodule...')
        if call(['make', 'install']) != 0:
            raise ValueError('Error during make')
        os.chdir('../..')
    
    # copy to desired location
    install_path = os.path.join(install_dir, 'lib','python' + '.'.join(sys.version.split('.')[:2]), 'site-packages')
    if not os.path.exists(install_path):
        os.makedirs(install_path)
    shutil.copytree('src/phocnet', install_path + '/phocnet')
    
    logger.info('Finished installation.')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for easy install of the PHOCNet library (dependencies must be present).')
    parser.add_argument('--cudnn-dir', type=str, help='Path to the CUDNN root dir.')
    parser.add_argument('--opencv-dir', type=str, help='Path to the OpenCV share dir.')
    parser.add_argument('--install-dir', type=str, required=True, help='Path to install the PHOCNet library into.')
    parser.add_argument('--install-caffe-dir', type=str, help='Path to install the custom Caffe library into. If unspecified, the install_ir path is chosen.')
    parser.add_argument('--no-caffe', action='store_true', 
                        help='If this flag is provided, the PHOCNet library is installed without the custom Caffe (e.g. if you installed a different Caffe version and don''')
    
    args = vars(parser.parse_args())
    main(**args)
