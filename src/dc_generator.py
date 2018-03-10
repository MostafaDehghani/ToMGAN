import tensorflow as tf


class Generator(object):
  def __init__(self, hps, scope='generator', depths=[1024, 512, 256, 128], s_size=4):
    self._hps = hps
    self._scope = scope
    self.initializer = tf.contrib.layers.xavier_initializer
    self.depths = depths + [3]
    self.s_size = s_size
    self._buid_generator_graph()

  def _buid_generator_graph(self):
    pass

  def generate(self, z, reuse=True, training=True):
    with tf.variable_scope(self._scope, reuse=reuse) as scope:
      # reshape from inputs
      with tf.variable_scope('reshape'):
        outputs = tf.layers.dense(z, self.depths[0] * self.s_size * self.s_size)
        outputs = tf.reshape(outputs, [-1, self.s_size, self.s_size, self.depths[0]])
        outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')

      # deconvolution (transpose of convolution) x 4
      with tf.variable_scope('deconv1'):
        outputs = tf.layers.conv2d_transpose(outputs, self.depths[1], [5, 5], strides=(2, 2),
                                             padding='SAME')
        outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
      with tf.variable_scope('deconv2'):
        outputs = tf.layers.conv2d_transpose(outputs, self.depths[2], [5, 5], strides=(2, 2),
                                             padding='SAME')
        outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
      with tf.variable_scope('deconv3'):
        outputs = tf.layers.conv2d_transpose(outputs, self.depths[3], [5, 5], strides=(2, 2),
                                             padding='SAME')
        outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
      with tf.variable_scope('deconv4'):
        outputs = tf.layers.conv2d_transpose(outputs, self.depths[4], [5, 5], strides=(2, 2),
                                             padding='SAME')
      # output images
      with tf.variable_scope('tanh'):
        outputs = tf.tanh(outputs, name='outputs')

    self._theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gan/" + self._scope)

    self.img = outputs

    return self.img
