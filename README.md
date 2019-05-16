## sequence-prediction

In development code for predicting image sequences using learned representation
of structure and motion. 

## Running on the cluster

To make sure you can run on the cluster in a default configuration, try the following:
```
kssh -i chaneyk/tensorflow:v1.5.0-py
python /PATH/TO/sequence_prediction/train_simple_rnn.py --train_batch_size 1 --val_batch_size 20 --base_dir /NAS/home/YOUR_USERNAME/logs/
```
This will train a simple CNN+RNN+CNN network for prediction on Moving MNIST. If things are running correctly, you'll see terminal output and you can monitor the network's outputs and training and validation performance with Tensorboard:
```
ktensorboard logs
```

### Dependencies
* Python 2.7
* [h5py](https://github.com/h5py/h5py) -- to save samples
* [Tensorflow](https://github.com/tensorflow/tensorflow/) version 1.5.
* [Sonnet](https://github.com/deepmind/sonnet) version 1.16.
* [Protobuf](https://github.com/google/protobuf)
* [OpenCV](https://opencv.org/)


### Datasets

* [Moving MNIST](https://github.com/emansim/unsupervised-videos)
* [Push Dataset](https://sites.google.com/site/brainrobotdata/home/push-dataset)
* [KTH Human Actions](http://www.nada.kth.se/cvap/actions/)
* [UCF101](http://crcv.ucf.edu/data/UCF101.php)

#####To work with Moving MNIST (have been set up for the daniilidis-group cluster):


- Download the dataset:
```
wget http://www.cs.toronto.edu/~emansim/datasets/mnist.h5
wget http://www.cs.toronto.edu/~emansim/datasets/bouncing_mnist_test.npy
```
 - change the value of *\_MNIST_DIR* in /moving_mnist/data_handler.py to the location of the dataset on your machine.
 -  change the value of *\_MNIST_TEST* in /configs.py to the location of the bouncing_mnist_test.npy file on your machine.

Further changes seem to be optional:

- To configure bouncing MNIST protobuf reading (the config_pb2 imports) run this in the /path/to/code/sequence_prediction/moving_mnist/ folder (where the config.proto file is):
```
sudo apt-get install protobuf-compiler
protoc -I=./ --python_out=./ config.proto
```

#####Others 
TBD after working on MNIST. KITTI, variants of moving MNIST with more factors of variation and
with change points?

### To run:

- In *base_dir* argument to train_simple_rnn.py pass the desired checkpoint storage folder, e.g.:

```
python code/sequence_prediction/train_simple_rnn.py -base_dir /NAS/home/logs/sequence_prediction/_current
```
