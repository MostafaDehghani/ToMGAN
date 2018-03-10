"""This file contains code to build and run the tensorflow graph
for the vanilla GAN  model"""

import time
# import lib
import tensorflow as tf
import numpy as np

from discriminator import Discriminator
from generator import Generator

FLAGS = tf.app.flags.FLAGS


class GAN_model(object):
  """"""

  def __init__(self, hps):
    self._hps = hps

  def _build_GAN(self):

    self.initializer = tf.contrib.layers.xavier_initializer
    self.discriminator_inner = Discriminator(self._hps, scope='discriminator_inner')
    self.discriminator = Discriminator(self._hps)
    self.generator = Generator(self._hps)

    with tf.variable_scope('gan'):
      # discriminator input from real data
      self._X = tf.placeholder(dtype=tf.float32, name='X',
                               shape=[None, self._hps.dis_input_size])
      # noise vector (generator input)
      self._Z = tf.placeholder(dtype=tf.float32, name='Z',
                               shape=[None, self._hps.gen_input_size])

      self.G_sample = self.generator.generate(self._Z)
      D_real, D_logit_real = self.discriminator.discriminate(self._X)
      D_fake, D_logit_fake = self.discriminator.discriminate(self.G_sample)
      D_in_fake, D_in_logit_fake = self.discriminator_inner.discriminate(
        self.G_sample)

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

    with tf.variable_scope('D_in_loss'):
      self._D_in_loss = tf.reduce_mean(tf.losses.mean_squared_error(
        predictions=D_in_logit_fake, labels=D_logit_fake))
      tf.summary.scalar('D_in_loss', self._D_in_loss, collections=['Dis_in'])

    with tf.variable_scope('G_loss'):
      self._G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                    (logits=D_in_logit_fake,
                                     labels=tf.ones_like(D_in_logit_fake)))
      tf.summary.scalar('G_loss', self._G_loss, collections=['Gen'])

  def _add_train_op(self):
    """Sets self._train_op, the op to run for training.
    """

    with tf.device("/gpu:0"):
      self._train_op_D = tf.train.AdamOptimizer().minimize(self._D_loss,
                                                           global_step=self.global_step_D,
                                                           var_list=self.discriminator._theta)
      self._train_op_D_in = tf.train.AdamOptimizer().minimize(self._D_in_loss,
                                                              global_step=self.global_step_D_in,
                                                              var_list=self.discriminator_inner._theta)

      self._train_op_G = tf.train.AdamOptimizer().minimize(self._G_loss,
                                                           global_step=self.global_step_G,
                                                           var_list=self.generator._theta)

  def build_graph(self):
    """Add the model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    with tf.device("/gpu:0"):
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

  def run_train_step(self, sess, batch, summary_writer, logging=False):
    """Runs one training iteration. Returns a dictionary containing train op,
    summaries, loss, global_step"""

    ######
    sample_Z = np.random.uniform(-1., 1., size=[self._hps.batch_size,
                                                self._hps.gen_input_size])
    feed_dict_D = {self._X: batch, self._Z: sample_Z}
    to_return_D = {
      'train_op': self._train_op_D,
      'summaries': self._summaries_D,
      'summaries_all': self._summaries_All,
      'loss': self._D_loss,
      'global_step_D': self.global_step_D,
      'global_step': self.global_step,
    }
    results_D = sess.run(to_return_D, feed_dict_D)

    ######

    sample_Z = np.random.uniform(-1., 1., size=[self._hps.batch_size,
                                                self._hps.gen_input_size])
    feed_dict_D_in = {self._X: batch, self._Z: sample_Z}
    to_return_D_in = {
      'train_op': self._train_op_D_in,
      'summaries': self._summaries_D_in,
      'summaries_all': self._summaries_All,
      'loss': self._D_in_loss,
      'global_step_D_in': self.global_step_D_in,
      'global_step': self.global_step,
    }
    results_D_in = sess.run(to_return_D_in, feed_dict_D_in)

    ######

    sample_Z = np.random.uniform(-1., 1., size=[self._hps.batch_size,
                                                self._hps.gen_input_size])
    feed_dict_G = {self._X: batch, self._Z: sample_Z}
    to_return_G = {
      'train_op': self._train_op_G,
      'summaries': self._summaries_G,
      'summaries_all': self._summaries_All,
      'loss': self._G_loss,
      'global_step_G': self.global_step_G,
      'global_step': self.global_step,

    }

    results_G = sess.run(to_return_G, feed_dict_G)

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

  def sample_generator(self, sess):
    """Runs generator to generate samples"""

    sample_Z = np.random.uniform(-1., 1., size=[16, self._hps.gen_input_size])
    feed_dict_G = {self._Z: sample_Z}
    to_return = {
      'g_sample': self.G_sample,
    }
    return sess.run(to_return, feed_dict_G)
