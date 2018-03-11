# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist

FLAGS = None


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  images = data_set.images
  labels = data_set.labels
  num_examples = data_set.num_examples

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  print(rows,cols,depth)
  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  with tf.python_io.TFRecordWriter(filename) as writer:
    for index in range(num_examples):
      image_raw = images[index].tostring()
      example = tf.train.Example(
        features=tf.train.Features(
          feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)
          }))
      writer.write(example.SerializeToString())


def load_tfrecord(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized,
    features={
      'label': tf.FixedLenFeature([], tf.string),
      'image': tf.FixedLenFeature([], tf.string)
    }
  )
  record_image = tf.decode_raw(features['image'], tf.uint8)

  image = tf.reshape(record_image, [500, 500, 1])
  label = tf.cast(features['label'], tf.string)
  min_after_dequeue = 10
  batch_size = 1000
  capacity = min_after_dequeue + 3 * batch_size
  image_batch, label_batch = tf.train.shuffle_batch(
    [image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue
  )

  return image_batch, label_batch


def main(unused_argv):
  """
  # Get the data.
  data_sets = mnist.read_data_sets(FLAGS.directory,
                                   dtype=tf.uint8,
                                   reshape=False,
                                   validation_size=FLAGS.validation_size)

  # Convert to Examples and write the result to TFRecords.
  convert_to(data_sets.train, 'train')
  convert_to(data_sets.validation, 'validation')
  convert_to(data_sets.test, 'test')
  """
  filename_queue = tf.train.string_input_producer(['/tmp/data/validation.tfrecords'], num_epochs=1)
  data = load_tfrecord(filename_queue)

  with tf.Session() as sess:
    print("running")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())
    Y = sess.run([data])
    print(Y)
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--directory',
    type=str,
    default='/tmp/data',
    help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
    '--validation_size',
    type=int,
    default=5000,
    help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
