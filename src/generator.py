import tensorflow as tf

class Generator(object):
    def __init__(self,hps):
        self._hps = hps
        self.initializer = tf.contrib.layers.xavier_initializer
        self._buid_generator_graph()


    def _buid_generator_graph(self):
        with tf.variable_scope('generator'):
            self.G_W1 = tf.get_variable(name="G_W1",
                                        shape=(self._hps.gen_input_size, self._hps.hidden_dim),
                                        initializer=self.initializer())
            self.G_b1 = tf.get_variable(name="G_b1",shape=(self._hps.hidden_dim),
                                        initializer=tf.zeros_initializer())

            self.G_W2 = tf.get_variable(name="G_W2",
                                        shape=(self._hps.hidden_dim, self._hps.gen_output_size),
                                        initializer=self.initializer())
            self.G_b2 = tf.get_variable(name="G_b2",
                                        shape=(self._hps.gen_output_size),
                                        initializer=tf.zeros_initializer())
            self._theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]


    def generate(self,z):
        self.G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1)
        self.G_logits = tf.matmul(self.G_h1, self.G_W2) + self.G_b2
        self.G_prob = tf.nn.sigmoid(self.G_logits)

        return self.G_prob


