import tensorflow as tf

class Generator(object):
    def __init__(self,hps, scope = 'generator'):
        self._hps = hps
        self._scope = scope
        self.initializer = tf.contrib.layers.xavier_initializer
        self._buid_generator_graph()


    def _buid_generator_graph(self):
        with tf.variable_scope(self._scope):
            self.W1 = tf.get_variable(name="W1",
                                      shape=(self._hps.gen_input_size, self._hps.hidden_dim),
                                      initializer=self.initializer())
            self.b1 = tf.get_variable(name="b1", shape=(self._hps.hidden_dim),
                                      initializer=tf.zeros_initializer())

            self.W2 = tf.get_variable(name="W2",
                                      shape=(self._hps.hidden_dim, self._hps.gen_output_size),
                                      initializer=self.initializer())
            self.b2 = tf.get_variable(name="b2",
                                      shape=(self._hps.gen_output_size),
                                      initializer=tf.zeros_initializer())
            self._theta = [self.W1, self.W2, self.b1, self.b2]


    def generate(self,z):
      with tf.variable_scope(self._scope):
        self.h1 = tf.nn.relu(tf.matmul(z, self.W1) + self.b1)
        self.logits = tf.matmul(self.h1, self.W2) + self.b2
        self.prob = tf.nn.sigmoid(self.logits)

      return self.prob


