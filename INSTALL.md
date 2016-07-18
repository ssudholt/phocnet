# Installation
## Customized Caffe
In order to run the experiment script you need to clone a customized Caffe version from Github and compile it from source:
```
git clone https://github.com/ssudholt/caffe
git checkout patrec-master
mkdir caffe/build
cd caffe/build
cmake ..
make
make install
```
If cmake exits with an error of unmet dependencies, make sure you installed all [Caffe dependencies](http://caffe.berkeleyvision.org/installation.html#prerequisites). The customized version does not introduce any new dependencies.
