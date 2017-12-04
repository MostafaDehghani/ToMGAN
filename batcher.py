from tensorflow.examples.tutorials.mnist import input_data
import os


class Batcher(object):

  def __init__(self, data_path, hps):
    self._hps = hps
    path = os.path.join(data_path,'MNIST_data')
    # if not os.path.exists(path):
    self._mnist = input_data.read_data_sets(path, one_hot=True)

  def next_batch(self):
    batch, _ = self._mnist.train.next_batch(self._hps.batch_size)
    return batch
