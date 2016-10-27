# Installation
The following installation instructions can be used for Debian- and Ubuntu-based systems. It is assumed that CMake and GCC are already installed on your system. If CUDA 8.0 or higher is installed and found in the `PATH`, the PHOCNet library is compiled with GPU support

## Install Dependency Packages
First, you need to install all dependency packages. For the underlying Caffe implementation, the depending packages can be installed by:
```
sudo apt-get install libprotobuf-dev libleveldb-dev liblmdb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev libgflags-dev libgoogle-glog-dev
sudo apt-get install --no-install-recommends libboost-all-dev
```
For the PHOCNet library, the following additional Python packages are needed:
```
sudo apt-get install python-opencv python-skimage python-sklearn python-scipy

```
Additionally, you need the Python LMDB binding package. In Ubuntu this package can be installed from the package manager as well (`python-lmdb`). For Debian Jessie and lower, you need to install it manually through `pip` or the [package index](https://pypi.python.org/pypi/lmdb).

If you wish to use CUDNN to speed up Caffe, you need to install it manually as well from [here](https://developer.nvidia.com/cudnn). This of course requires CUDA.

## install.py
You can now use the supplied `install.py` script to install the PHOCNet library. In the following, we will refer to the directory you want to install the library into as `<PHOCNet install dir>` Change to the directory you cloned the Github repository into and run:
```
python install.py --install-dir <PHOCNet install dir>
```
If you are using CUDNN, you can supply its root directory as parameter as well for the PHOCNet library to use it:
```
python install.py --install-dir <PHOCNet install dir> --cudnn-dir <path to your CUDNN dir>
```
In both cases, the custom Caffe version supplied with the PHOCNet library is installed into `<PHOCNet install dir>`. You can change this default behavior throught the `--install-caffe-dir` argument:
```
python install.py --install-dir <PHOCNet install dir> --install-caffe-dir <Caffe install dir>
```
There exist a number of advanced options to further customize the installation process. If you're interested in them, you can find out more usage information of the installation script through
```
python install.py -h
```

## Environment Variables
Everything is setup now. For the library to work, all that is left to do is to add the `<PHOCNet install dir>/caffe/lib` directory to the `LD_BIRARY_PATH` and the `<PHOCNet install dir>/caffe/python` as well as the `<Caffe install dir>/lib/python2.7/site-packages/` to your `PYTHONPATH`.
