"""This file contains code to build and run the tensorflow graph
for the vanilla GAN  model"""

import time
import lib
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS


class TomGAN_model(object):
  """"""

  def __init__(self, hps):
    self._hps = hps

  def _build_GAN(self):
    """Add the whole GAN  model to the graph."""
    with tf.variable_scope('discriminator'):
      D_W1 = tf.Variable(lib.xavier_init([self._hps.dis_input_size, self._hps.hidden_dim]))
      D_b1 = tf.Variable(tf.zeros(shape=[self._hps.hidden_dim]))

      D_W2 = tf.Variable(lib.xavier_init([self._hps.hidden_dim, 1]))
      D_b2 = tf.Variable(tf.zeros(shape=[1]))
      self._theta_D = [D_W1, D_W2, D_b1, D_b2]

      def discriminator(x):
        D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
        D_logit = tf.matmul(D_h1, D_W2) + D_b2
        D_prob = tf.nn.sigmoid(D_logit)

        return D_prob, D_logit

    with tf.variable_scope('generator'):
      G_W1 = tf.Variable(
        lib.xavier_init([self._hps.gen_input_size, self._hps.hidden_dim]))
      G_b1 = tf.Variable(tf.zeros(shape=[self._hps.hidden_dim]))

      G_W2 = tf.Variable(lib.xavier_init([self._hps.hidden_dim,
                                          self._hps.dis_input_size ]))
      G_b2 = tf.Variable(tf.zeros(shape=[self._hps.dis_input_size]))
      self._theta_G = [G_W1, G_W2, G_b1, G_b2]

      def generator(z):
        G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
        G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)

        return G_prob

    with tf.variable_scope('gan'):

      # discriminator input from real data
      self._X = tf.placeholder(dtype = tf.float32, name= 'X',
                               shape=[None, self._hps.dis_input_size])
      # noise vector (generator input)
      self._Z = tf.placeholder(dtype = tf.float32, name= 'Z',
        shape=[None,self._hps.gen_input_size ])

      self.G_sample = generator(self._Z)
      D_real, D_logit_real = discriminator(self._X)
      D_fake, D_logit_fake = discriminator(self.G_sample)


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
      tf.summary.scalar('D_loss_real', D_loss_real)
      tf.summary.scalar('D_loss_fake', D_loss_fake)
      tf.summary.scalar('D_loss', self._D_loss)


    with tf.variable_scope('G_loss'):
      self._G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                    (logits=D_logit_fake,
                                     labels=tf.ones_like(D_logit_fake)))
      tf.summary.scalar('G_loss', self._G_loss)


  def _add_train_op(self):
    """Sets self._train_op, the op to run for training.
    """

    with tf.device("/gpu:0"):
      self._train_op_D = tf.train.AdamOptimizer().minimize(self._D_loss,
                                                  global_step=self.global_step,
                                                         var_list=self._theta_D)
      self._train_op_G = tf.train.AdamOptimizer().minimize(self._G_loss,
                                                  global_step=self.global_step,
                                                         var_list=self._theta_G)

    # Alternative: Mode control over optimization hyperparameters
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
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self._add_train_op()
    self._summaries = tf.summary.merge_all()
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)



  def run_train_step(self, sess, batch):
    """Runs one training iteration. Returns a dictionary containing train op,
    summaries, loss, global_step"""

    sample_Z = np.random.uniform(-1., 1., size=[self._hps.batch_size,
                                                self._hps.gen_input_size])
    feed_dict_D = {self._X: batch, self._Z: sample_Z}
    to_return_D = {
        'train_op': self._train_op_D,
        'summaries': self._summaries,
        'loss': self._D_loss,
        'global_step': self.global_step,
    }
    results_D = sess.run(to_return_D, feed_dict_D)

    sample_Z = np.random.uniform(-1., 1., size=[self._hps.batch_size,
                                                 self._hps.gen_input_size])
    feed_dict_G = {self._X: batch, self._Z: sample_Z}
    to_return_G = {
        'train_op': self._train_op_G,
        'summaries': self._summaries,
        'loss': self._G_loss,
        'global_step': self.global_step,
    }

    results_G = sess.run(to_return_G, feed_dict_G)

    to_return = {
      'loss_D': results_D['loss'],
      'loss_G': results_G['loss'],
      'summaries': results_G['summaries'],
      'global_step': results_G['global_step'],
    }

    return to_return

  def sample_generator(self, sess):
    """Runs generator to generate samples"""

    sample_Z = np.random.uniform(-1., 1., size=[16, self._hps.gen_input_size])
    feed_dict_G = {self._Z: sample_Z}
    to_return = {
        'g_sample': self.G_sample,
    }
    return sess.run(to_return, feed_dict_G)
