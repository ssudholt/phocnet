# PHOCNet

PHOCNet is a Convolutional Neural Network for classifying document image attributes. This code was the base for generating the results in [PHOCNet: A Deep Convolutional Neural Network for Word Spotting in Handwritten Documents](https://arxiv.org/abs/1604.00187)

## Prerequisites
In order to use the PHOCNet library, you need to install the following depenencies:
- [Caffe](https://github.com/BVLC/caffe)
- numpy
- skimage
- scipy
- LMDB/Python LMDB Bindings

## Usage
You can either embed this code in your project and call the class from there or use the experiment script `experiments/phocnet_experiment.py`.
Usage information are provided through
```
python phocnet_experiment.py
```
This command also gives an overview over the possible parameters and their default values.

The minimally required parameters for running a PHOCNet experiment are
```
python phocnet_experiment.py --doc_img_dir <folder of the doc. images> --train_annnotation_file <READ-style train XML> --test_annotation_file <READ-style test XML> --proto_dir <folder for protofiles> --lmdb_dir <folder for lmdbs>
```

### READ-style XMLs
PHOCNet experiments need READ-style XML files for the training and test partitions of the individual datasets. The layout of such an XML file is shown below.
    <?xml version="1.0" encoding="utf-8" ?>
    <wordLocations dataset="my_dataset">
        <spot word="convolutional" image="doc_image1" x="123" y="55" w="123" h="50" />
        <spot word="neural" image="doc_image1" x="553" y="97" w="100" h="59" />
        <spot word="networks" image="doc_image2" x="94" y="1197" w="244" h="62" />
        <!-- The rest of the words in the dataset -->
    </wordLocations>
A number of sample XML files can be found under `experiments`. You can either use the sample XML files or create your own training and test XML files.

### LMDB
For fast training, the PHOCNet library makes use of LMDB database as input for Caffe. During the first run of an experiment, the LMDBs are created automatically. For this you need to specify where to save the LMDB files. Keep in mind that LMDBs can easily grow to a couple of GBs.

