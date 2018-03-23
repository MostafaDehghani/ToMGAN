"""This file contains code to build and run the tensorflow graph
for the vanilla GAN  model"""

import time
# import lib
import tensorflow as tf
import numpy as np
import os

from tensorflow.examples.tutorials.mnist import mnist

from discriminator import Discriminator
from generator import Generator

import sys
sys.path.append(os.path.join('..', '..', 'models-master', 'research', 'gan'))
from mnist import util

FLAGS = tf.app.flags.FLAGS


class GAN_model(object):
  """"""

  def __init__(self, hps, s_size=4):
    self._hps = hps
    self.s_size = s_size

  def inputs(self, batch_size, s_size):
    files = [os.path.join(self._hps.data_path, f) for f in os.listdir(self._hps.data_path) if f.endswith('.tfrecords')]
    print("tfrecord files: ", files)
    fqueue = tf.train.string_input_producer(files)
    reader = tf.TFRecordReader()
    _, value = reader.read(fqueue)
    features = tf.parse_single_example(value, features={'image_raw': tf.FixedLenFeature([], tf.string),
                                                        'height': tf.FixedLenFeature([], tf.int64),
                                                        'width': tf.FixedLenFeature([], tf.int64),
                                                        'depth': tf.FixedLenFeature([], tf.int64)})
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape((mnist.IMAGE_PIXELS))
    image = tf.cast(image, tf.float32) * (1 / 255)
    #image = tf.image.resize_image_with_crop_or_pad(image, CROP_IMAGE_SIZE, CROP_IMAGE_SIZE)
    #image = tf.image.random_flip_left_right(image)

    min_queue_examples = self._hps.batch_size * 2
    images = tf.train.shuffle_batch(
      [image],
      batch_size=batch_size,
      capacity=min_queue_examples + 3 * batch_size,
      min_after_dequeue=min_queue_examples)
    tf.summary.image('images', images)

    return images #.resize_images(images, [s_size * 2 ** 4, s_size * 2 ** 4])

  def _build_GAN(self):

    self.initializer = tf.contrib.layers.xavier_initializer


    with tf.variable_scope('gan'):
      # discriminator input from real data
      self._X = self.inputs(self._hps.batch_size, self.s_size)
      # tf.placeholder(dtype=tf.float32, name='X',
      #                       shape=[None, self._hps.dis_input_size])
      # noise vector (generator input)
      self._preZ = tf.random_uniform([self._hps.batch_size*3, self._hps.gen_input_size], minval=-1.0, maxval=1.0)
      self._Z = tf.random_uniform([self._hps.batch_size, self._hps.gen_input_size], minval=-1.0, maxval=1.0)
      self._Z_sample = tf.random_uniform([20, self._hps.gen_input_size], minval=-1.0, maxval=1.0)


      self.discriminator_inner = Discriminator(self._hps, scope='discriminator_inner')
      self.discriminator = Discriminator(self._hps)
      self.generator = Generator(self._hps)

      # Generator
      self.G_presample = self.generator.generate(self._preZ)
      self.G_sample_test = self.generator.generate(self._Z_sample)

      # Inner Discriminator
      D_in_fake_presample, D_in_logit_fake_presample = self.discriminator_inner.discriminate(self.G_presample)
      D_in_real, D_in_logit_real = self.discriminator_inner.discriminate(self._X)

      values, indices =  tf.nn.top_k(D_in_fake_presample[:,0],self._hps.batch_size)
      tf.logging.info(indices)
      self.G_selected_samples = tf.gather(self.G_presample,indices)
      tf.logging.info(self.G_selected_samples)


      D_in_fake, D_in_logit_fake = self.discriminator_inner.discriminate(self.G_selected_samples)

      # Discriminator
      D_real, D_logit_real = self.discriminator.discriminate(self._X)
      D_fake, D_logit_fake = self.discriminator.discriminate(self.G_selected_samples)




    with tf.variable_scope('D_loss'):
      D_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real,
                                                labels=tf.ones_like(
                                                  D_logit_real)))
      D_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                labels=tf.zeros_like(
                                                  D_logit_fake)))
      self._D_loss = D_loss_real + D_loss_fake
      tf.summary.scalar('D_loss_real', D_loss_real, collections=['Dis'])
      tf.summary.scalar('D_loss_fake', D_loss_fake, collections=['Dis'])
      tf.summary.scalar('D_loss', self._D_loss, collections=['Dis'])
      tf.summary.scalar('D_out', tf.reduce_mean(D_logit_fake), collections=['Dis'])

    with tf.variable_scope('D_in_loss'):
      D_in_loss_fake = tf.reduce_mean(tf.losses.mean_squared_error(
        predictions=D_in_logit_fake, labels=D_logit_fake))
      D_in_loss_real = tf.reduce_mean(tf.losses.mean_squared_error(
        predictions=D_in_logit_real, labels=D_logit_real))
      self._D_in_loss = D_in_loss_fake + D_in_loss_real
      tf.summary.scalar('D_in_loss', self._D_in_loss, collections=['Dis_in'])
      tf.summary.scalar('D_in_out', tf.reduce_mean(D_in_logit_fake), collections=['Dis_in'])

    with tf.variable_scope('G_loss'):
      self._G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                    (logits=D_in_logit_fake,
                                     labels=tf.ones_like(D_in_logit_fake)))
      tf.summary.scalar('G_loss', self._G_loss, collections=['Gen'])


    with tf.variable_scope('GAN_Eval'):
      MNIST_CLASSIFIER_FROZEN_GRAPH = '../../models-master/research/gan/mnist/data/classify_mnist_graph_def.pb'
      tf.logging.info(self.G_sample_test.shape)
      eval_fake_images = tf.image.resize_images(self.G_sample_test,[28,28])
      eval_real_images = tf.image.resize_images(self._X[:20],[28,28])
      self.eval_score = util.mnist_score(eval_fake_images, MNIST_CLASSIFIER_FROZEN_GRAPH)
      self.frechet_distance = util.mnist_frechet_distance(
        eval_real_images, eval_fake_images, MNIST_CLASSIFIER_FROZEN_GRAPH)

      tf.summary.scalar('MNIST_Score',self.eval_score,collections=['All'])
      tf.summary.scalar('frechet_distance', self.frechet_distance, collections=['All'])

  def _add_train_op(self):
    """Sets self._train_op, the op to run for training.
    """
    with tf.device("/gpu:0"):
      learning_rate_D = 0.0004  # tf.train.exponential_decay(0.001, self.global_step_D,
      #                                           100000, 0.96, staircase=True)
      learning_rate_G = 0.0004  # tf.train.exponential_decay(0.001, self.global_step_G,
      #                                           100000, 0.96, staircase=True)
      learning_rate_D_in = 0.0004  # tf.train.exponential_decay(0.001, self.global_step_D,
      #                                           100000, 0.96, staircase=True)
      self._train_op_D = tf.train.AdamOptimizer(learning_rate_D,beta1=0.5).minimize(self._D_loss,
                                                           global_step=self.global_step_D,
                                                           var_list=self.discriminator._theta)
      self._train_op_D_in = tf.train.AdamOptimizer(learning_rate_D_in,beta1=0.5).minimize(self._D_in_loss,
                                                              global_step=self.global_step_D_in,
                                                              var_list=self.discriminator_inner._theta)

      self._train_op_G = tf.train.AdamOptimizer(learning_rate_G,beta1=0.5).minimize(self._G_loss,
                                                           global_step=self.global_step_G,
                                                           var_list=self.generator._theta)

  def build_graph(self):
    """Add the model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    # with tf.device("/gpu:0"):
    self._build_GAN()

    self.global_step_D = tf.Variable(0, name='global_step_D', trainable=False)
    self.global_step_D_in = tf.Variable(0, name='global_step_D_in',
                                        trainable=False)
    self.global_step_G = tf.Variable(0, name='global_step_G', trainable=False)
    self.global_step = tf.add(tf.add(self.global_step_G, self.global_step_D),
                              self.global_step_D_in, name='global_step')

    tf.summary.scalar('global_step_D', self.global_step_D, collections=['All'])
    tf.summary.scalar('global_step_D_in', self.global_step_D_in,
                      collections=['All'])
    tf.summary.scalar('global_step_G', self.global_step_G, collections=['All'])
    self._add_train_op()
    self._summaries_D = tf.summary.merge_all(key='Dis')
    self._summaries_D_in = tf.summary.merge_all(key='Dis_in')
    self._summaries_G = tf.summary.merge_all(key='Gen')
    self._summaries_All = tf.summary.merge_all(key='All')
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)

  def run_train_step(self, sess, summary_writer, logging=False):
    """Runs one training iteration. Returns a dictionary containing train op,
    summaries, loss, global_step"""

    ######
    to_return_D = {
      'train_op': self._train_op_D,
      'summaries': self._summaries_D,
      'summaries_all': self._summaries_All,
      'loss': self._D_loss,
      'global_step_D': self.global_step_D,
      'global_step': self.global_step,
    }
    results_D = sess.run(to_return_D)

    ######

    to_return_D_in = {
      'train_op': self._train_op_D_in,
      'summaries': self._summaries_D_in,
      'summaries_all': self._summaries_All,
      'loss': self._D_in_loss,
      'global_step_D_in': self.global_step_D_in,
      'global_step': self.global_step,
    }
    results_D_in = sess.run(to_return_D_in)

    ######

    to_return_G = {
      'train_op': self._train_op_G,
      'summaries': self._summaries_G,
      'summaries_all': self._summaries_All,
      'loss': self._G_loss,
      'global_step_G': self.global_step_G,
      'global_step': self.global_step,

    }

    results_G = sess.run(to_return_G)

    # we will write these summaries to tensorboard using summary_writer
    summaries_G = results_G['summaries']
    summaries_D = results_D['summaries']
    summaries_D_in = results_D_in['summaries']
    summaries_All = results_G['summaries_all']

    global_step_G = results_G['global_step_G']
    global_step_D = results_D['global_step_D']
    global_step_D_in = results_D_in['global_step_D_in']
    global_step = results_G['global_step']

    summary_writer.add_summary(summaries_G,
                               global_step_G)  # write the summaries
    summary_writer.add_summary(summaries_D,
                               global_step_D)  # write the summaries
    summary_writer.add_summary(summaries_D_in,
                               global_step_D_in)  # write the summaries
    summary_writer.add_summary(summaries_All,
                               global_step)  # write the summaries

    if logging:

      loss_D = results_D['loss']
      tf.logging.info('loss_D: %f', loss_D)  # print the loss to screen

      loss_D_in = results_D_in['loss']
      tf.logging.info('loss_D_in: %f', loss_D_in)  # print the loss to screen

      loss_G = results_G['loss']
      tf.logging.info('loss_G: %f', loss_G)  # print the loss to screen

      if not np.isfinite(loss_G):
        raise Exception("Loss_G is not finite. Stopping.")
      if not np.isfinite(loss_D):
        raise Exception("Loss_D is not finite. Stopping.")
      if not np.isfinite(loss_D_in):
        raise Exception("Loss_D_in is not finite. Stopping.")

      # flush the summary writer every so often
      summary_writer.flush()


  def run_eval_step(self,sess):

    return sess.run([self.eval_score,self.frechet_distance])


  def sample_generator(self, sess):
    """Runs generator to generate samples"""

    to_return = {
      'g_sample': self.G_sample_test,
    }
    return sess.run(to_return)
