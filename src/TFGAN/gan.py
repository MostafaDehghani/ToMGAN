from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import time
import functools
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

# Main TFGAN library.
tfgan = tf.contrib.gan

# TFGAN MNIST examples from `tensorflow/models`.
from mnist import data_provider
from mnist import util

# TF-Slim data provider.
from datasets import download_and_convert_mnist

# Shortcuts for later.
queues = tf.contrib.slim.queues
layers = tf.contrib.layers
ds = tf.contrib.distributions
framework = tf.contrib.framework

leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)


def visualize_training_generator(train_step_num, start_time, data_np):
  """Visualize generator outputs during training.

  Args:
      train_step_num: The training step number. A python integer.
      start_time: Time when training started. The output of `time.time()`. A
          python float.
      data: Data to plot. A numpy array, most likely from an evaluated
      TensorFlow
          tensor.
  """
  print('Training step: %i' % train_step_num)
  time_since_start = (time.time() - start_time) / 60.0
  print('Time since start: %f m' % time_since_start)
  print('Steps per min: %f' % (train_step_num / time_since_start))
  plt.axis('off')
  plt.imshow(np.squeeze(data_np), cmap='gray')
  plt.show()


def visualize_digits(tensor_to_visualize):
  """Visualize an image once. Used to visualize generator before training.

  Args:
      tensor_to_visualize: An image tensor to visualize. A python Tensor.
  """
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    with queues.QueueRunners(sess):
      images_np = sess.run(tensor_to_visualize)
  plt.axis('off')
  plt.imshow(np.squeeze(images_np), cmap='gray')


def evaluate_tfgan_loss(gan_loss, name=None):
  """Evaluate GAN losses. Used to check that the graph is correct.

  Args:
      gan_loss: A GANLoss tuple.
      name: Optional. If present, append to debug output.
  """
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    with queues.QueueRunners(sess):
      gen_loss_np = sess.run(gan_loss.generator_loss)
      dis_loss_np = sess.run(gan_loss.discriminator_loss)
  if name:
    print('%s generator loss: %f' % (name, gen_loss_np))
    print('%s discriminator loss: %f' % (name, dis_loss_np))
  else:
    print('Generator loss: %f' % gen_loss_np)
    print('Discriminator loss: %f' % dis_loss_np)