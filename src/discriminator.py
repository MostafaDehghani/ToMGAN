import tensorflow as tf

class Discriminator(object):
    def __init__(self,hps):
        self._hps = hps
        self.initializer = tf.contrib.layers.xavier_initializer()
        self._buid_discriminator_graph()


    def _buid_discriminator_graph(self):
        """Add the whole GAN  model to the graph."""
        with tf.variable_scope('discriminator'):
            self.D_W1 = tf.get_variable(name="D_W1",
                                        shape=(self._hps.dis_input_size, self._hps.hidden_dim),
                                        initializer=self.initializer)
            self.D_b1 = tf.get_variable(name="D_b1",
                                        shape=(self._hps.hidden_dim),
                                        initializer=tf.zeros_initializer())

            self.D_W2 = tf.get_variable(name="D_W2",  shape=(self._hps.hidden_dim, self._hps.dis_output_size),
                                        initializer=self.initializer)
            self.D_b2 = tf.get_variable(name="D_b2",  shape=(self._hps.dis_output_size),
                                        initializer=tf.zeros_initializer())
            self._theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]



    def discriminate(self,x):
        self.D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
        self.D_logit = tf.matmul(self.D_h1, self.D_W2) + self.D_b2
        self.D_prob = tf.nn.sigmoid(self.D_logit)

        return self.D_logit,self.D_prob


