"""This file contains code to build and run the tensorflow graph
for the vanilla GAN  model"""

import time
#import lib
import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import mnist
import time
from discriminator import Discriminator
from generator import Generator




class GAN_model(object):
  """"""

  def __init__(self, hps, s_size=4):
    self._hps = hps
    self.s_size = s_size
    #tf.set_random_seed(np.random.get_state())


  def inputs(self, batch_size, s_size):
    files = [os.path.join(self._hps.data_path, f) for f in os.listdir(self._hps.data_path) if f.endswith('.tfrecords')]
    print("tfrecord files: ", files)
    fqueue = tf.train.string_input_producer(files)
    reader = tf.TFRecordReader()
    _, value = reader.read(fqueue)
    features = tf.parse_single_example(value, features={
                                                        'height': tf.FixedLenFeature([], tf.int64),
                                                        'width': tf.FixedLenFeature([], tf.int64),
                                                        'depth': tf.FixedLenFeature([], tf.int64),
                                                        'image_raw': tf.FixedLenFeature([], tf.string)})
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape((mnist.IMAGE_PIXELS))
    image = tf.cast(image, tf.float32) * (1 / 255)

    min_queue_examples = self._hps.batch_size * 2
    images = tf.train.shuffle_batch(
      [image],
      batch_size=batch_size,
      capacity=min_queue_examples + 3 * batch_size,
      min_after_dequeue=min_queue_examples)
    tf.summary.image('images', images)

    return images
    #return tf.image.resize_images(images, [s_size * 2 ** 4, s_size * 2 ** 4])


  def _build_GAN(self):

    self.initializer = tf.contrib.layers.xavier_initializer
    self.discriminator = Discriminator(self._hps)
    self.generator = Generator(self._hps)

    with tf.variable_scope('gan'):
      # discriminator input from real data
      image_input = self.inputs(self._hps.batch_size, self.s_size)
      tf.logging.info("image input")
      tf.logging.info(image_input)
      self._X = image_input #tf.contrib.layers.flatten(image_input)

      # tf.placeholder(dtype=tf.float32, name='X',
      #                       shape=[None, self._hps.dis_input_size])
      # noise vector (generator input)
      self._Z = tf.placeholder(dtype="float32", name='Z',shape=[None,self._hps.gen_input_size])
      #tf.random_uniform([self._hps.batch_size, self._hps.gen_input_size], minval=-1.0, maxval=1.0)
      #self._Z_sample = tf.random_uniform([20, self._hps.gen_input_size], minval=-1.0, maxval=1.0)

      # Generator
      self.G_sample = self.generator.generate(self._Z)

      D_real, D_logit_real = self.discriminator.discriminate(self._X)
      D_fake, D_logit_fake = self.discriminator.discriminate(self.G_sample)


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
      tf.summary.scalar('D_loss_fake', D_loss_fake,collections=['Dis'])
      tf.summary.scalar('D_loss', self._D_loss,collections=['Dis'])


    with tf.variable_scope('G_loss'):
      self._G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                    (logits=D_logit_fake,
                                     labels=tf.ones_like(D_logit_fake)))
      tf.summary.scalar('G_loss', self._G_loss,collections=['Gen'])


  def _add_train_op(self):
    """Sets self._train_op, the op to run for training.
    """

    with tf.device("/gpu:0"):
      tf.logging.info(self.discriminator._theta)
      learning_rate_D = 0.0004 #tf.train.exponential_decay(0.001, self.global_step_D,
      #                                           100000, 0.96, staircase=True)
      learning_rate_G = 0.0004 # tf.train.exponential_decay(0.001, self.global_step_G,
      #                                           100000, 0.96, staircase=True)
      #learning_rate_D,beta1=0.5
      self._train_op_D = tf.train.AdamOptimizer(learning_rate_D,beta1=0.5).minimize(self._D_loss,
                                                           global_step=self.global_step_D,
                                                           var_list=self.discriminator._theta)
      tf.logging.info(self.generator._theta)
      #learning_rate_G,beta1=0.5
      self._train_op_G = tf.train.AdamOptimizer(learning_rate_G,beta1=0.5).minimize(self._G_loss,
                                                           global_step=self.global_step_G,
                                                           var_list=self.generator._theta)

    # Alternative: More control over optimization hyperparameters
    # # Take gradients of the trainable variables w.r.t. the loss function to minimize
    # gradients_D = tf.gradients(self._D_loss, self._theta_G,
    #                          aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
    # gradients_G = tf.gradients(self._D_loss, self._theta_D,
    #                          aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
    #
    # # Clip the gradients
    # with tf.device("/gpu:0"):
    #   grads_G, global_norm_G = tf.clip_by_global_norm(gradients_G,
    #                                                   self._hps.max_grad_norm)
    #   grads_D, global_norm_D = tf.clip_by_global_norm(gradients_D,
    #                                                   self._hps.max_grad_norm)
    #
    # # Add a summary
    # tf.summary.scalar('global_norm_D', global_norm_D)
    # tf.summary.scalar('global_norm_G', global_norm_G)
    #
    # # Apply adagrad optimizer
    # optimizer = tf.train.AdagradOptimizer(self._hps.lr,
    #                                       initial_accumulator_value=self._hps.adagrad_init_acc)
    # with tf.device("/gpu:0"):
    #   self._train_op = optimizer.apply_gradients(zip(grads_D, self._theta_D),
    #                                              global_step=self.global_step,
    #                                              name='train_step')
    #   self._train_op = optimizer.apply_gradients(zip(grads_G, self._theta_G),
    #                                              global_step=self.global_step,
    #                                              name='train_step')


  def build_graph(self):
    """Add the model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    with tf.device("/gpu:0"):
      self._build_GAN()
    self.global_step_D = tf.Variable(0, name='global_step_D', trainable=False)
    self.global_step_G = tf.Variable(0, name='global_step_G', trainable=False)
    self.global_step = tf.add(self.global_step_G, self.global_step_D, name='global_step')

    tf.summary.scalar('global_step_D', self.global_step_D,collections=['All'])
    tf.summary.scalar('global_step_G', self.global_step_G,collections=['All'])
    self._add_train_op()
    self._summaries_D = tf.summary.merge_all(key='Dis')
    self._summaries_G = tf.summary.merge_all(key='Gen')
    self._summaries_All = tf.summary.merge_all(key='All')
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)



  def run_train_step(self, sess, summary_writer, logging = False):
    """Runs one training iteration. Returns a dictionary containing train op,
    summaries, loss, global_step"""
    feed_dict ={self._Z: np.random.uniform(-1,1,size=[20,self._hps.gen_input_size])}

    to_return_D = {
        'train_op': self._train_op_D,
        'summaries': self._summaries_D,
        'summaries_all': self._summaries_All,
        'loss': self._D_loss,
        'global_step_D': self.global_step_D,
        'global_step': self.global_step,
    }
    results_D = sess.run(to_return_D,feed_dict=feed_dict)


    to_return_G = {
        'train_op': self._train_op_G,
        'summaries': self._summaries_G,
        'summaries_all': self._summaries_All,
        'loss': self._G_loss,
        'global_step_G': self.global_step_G,
        'global_step': self.global_step,

    }
    results_G = sess.run(to_return_G,feed_dict=feed_dict)

    # we will write these summaries to tensorboard using summary_writer
    summaries_G = results_G['summaries']
    summaries_D = results_D['summaries']
    summaries_All = results_G['summaries_all']

    global_step_G = results_G['global_step_G']
    global_step_D = results_D['global_step_D']
    global_step = results_G['global_step']

    summary_writer.add_summary(summaries_G, global_step_G)  # write the summaries
    summary_writer.add_summary(summaries_D, global_step_D)  # write the summaries
    summary_writer.add_summary(summaries_All, global_step)  # write the summaries


    if logging:

        loss_D = results_D['loss']
        tf.logging.info('loss_D: %f', loss_D)  # print the loss to screen

        loss_G = results_G['loss']
        tf.logging.info('loss_G: %f', loss_G)  # print the loss to screen

        if not np.isfinite(loss_G):
            raise Exception("Loss_G is not finite. Stopping.")
        if not np.isfinite(loss_D):
            raise Exception("Loss_D is not finite. Stopping.")

        # flush the summary writer every so often
        summary_writer.flush()


  def sample_generator(self, sess):
    """Runs generator to generate samples"""

    to_return = {
        'g_sample': self.G_sample,
    }
    feed_dic ={self._Z: np.random.uniform(-1,1,size=[20,self._hps.gen_input_size])}
    return sess.run(to_return,feed_dic)
