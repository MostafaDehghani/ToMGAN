import os
import time
from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import util

FLAGS = tf.app.flags.FLAGS

# Where to save output
tf.app.flags.DEFINE_string('log_root', 'log_root', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('data_path', '../data/MNIST_data', 'Directory where the data '
                                                              'is going to be saved.')
tf.app.flags.DEFINE_string('dataset_id', 'cifar', 'cifar/mnist')
tf.app.flags.DEFINE_string('exp_name', 'main_experiment', 'Name for experiment. Logs will '
                                                          'be saved in a directory with this'
                                                          ' name, under log_root.')
tf.app.flags.DEFINE_string('model', 'vanilla', 'must be one of '
                                               'vanilla/ToM_cycle')
tf.app.flags.DEFINE_string('loss', 'normal', 'normal/wasserstein')
tf.app.flags.DEFINE_integer('hidden_dim', 128, 'dimension of hidden states '
                                               'in discriminator and generator')
tf.app.flags.DEFINE_integer('batch_size', 64, 'minibatch size')

tf.app.flags.DEFINE_integer('dis_output_size', 1, 'size of the input for the '
                                                  'discriminator (1)')

tf.app.flags.DEFINE_integer('dis_input_size', 784, 'size of the input for the '
                                                   'discriminator (for mnist = 784)')

tf.app.flags.DEFINE_integer('gen_input_size', 100, 'size of the noise vector '
                                                   'as the input for the generator')
tf.app.flags.DEFINE_integer('gen_output_size', 784, 'size of the generator output vector')
tf.app.flags.DEFINE_integer('logging_step', 500, 'logging step')
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_train', 5000,
                            """number of examples for train""")
# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.app.flags.DEFINE_boolean('debug', False,
                            "Run in tensorflow's debug mode (watches for NaN/inf values)")


# def restore_last_model():
#   exit()


def setup_training(model):
  """Does setup before starting training (run_training)"""
  train_dir = os.path.join(FLAGS.log_root, "train")
  if not os.path.exists(train_dir): os.makedirs(train_dir)

  model.build_graph()  # build the graph
  # if FLAGS.restore_last_model:
  #   restore_last_model()
  saver = tf.train.Saver(max_to_keep=3)  # keep 3 checkpoints at a time

  sv = tf.train.Supervisor(logdir=train_dir,
                           is_chief=True,
                           saver=saver,
                           summary_op=None,
                           save_summaries_secs=60,  # save summaries for tensorboard every 60 secs
                           save_model_secs=60,  # checkpoint every 60 secs
                           global_step=None
                           )

  summary_writer = sv.summary_writer
  tf.logging.info("Preparing or waiting for session...")
  sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
  tf.logging.info("Created session.")
  try:
    run_training(model, sess_context_manager, summary_writer)
    # this is an infinite loop until interrupted
  except KeyboardInterrupt:
    tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
    sv.stop()


def run_training(model, sess_context_manager, summary_writer):
  """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
  tf.logging.info("starting run_training")
  with sess_context_manager as sess:
    tf.train.start_queue_runners(sess=sess)
    if FLAGS.debug:  # start the tensorflow debugger
      sess = tf_debug.LocalCLIDebugWrapperSession(sess)
      sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    num_batch = 0
    results = model.sample_input(sess)
    print("image min:",np.min(results['input_images']))
    print("image max:",np.max(results['input_images']))
    util.plot(results['input_images'][:20], 0, 3)

    while True:  # repeats until interrupted
      if num_batch % FLAGS.logging_step == 0:
        tf.logging.info('------ number of  batches: ' + str(num_batch) + ' ------')
        t0 = time.time()
        model.run_train_step(sess, summary_writer, logging=True)
        t1 = time.time()
        tf.logging.info('seconds for training step: %.3f', t1 - t0)
        tf.logging.info("sampling from the generator")
        sampling_result = model.sample_generator(sess)
        if FLAGS.dataset_id == "mnist":
          util.plot(sampling_result['g_sample'], num_batch,1)
        elif FLAGS.dataset_id == "cifar":
          util.plot(sampling_result['g_sample'], num_batch, 3)
        print(model.run_eval_step(sess))
      else:  # no logging
        model.run_train_step(sess, summary_writer, logging=False)

      num_batch += 1


def main(unused_argv):
  # if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
  #   raise Exception("Problem with flags: %s" % unused_argv)

  tf.logging.set_verbosity(tf.logging.INFO)  # choose what level of logging you want

  # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
  FLAGS.log_root = os.path.join(FLAGS.log_root,FLAGS.dataset_id, FLAGS.model, FLAGS.exp_name)
  if not os.path.exists(FLAGS.log_root):
    os.makedirs(FLAGS.log_root)

  # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
  hparam_list = ['batch_size', 'hidden_dim', 'dis_input_size', 'gen_input_size',
                 'dis_output_size', 'gen_output_size', 'data_path','dataset_id','num_examples_per_epoch_for_train','loss']
  hps_dict = {}
  for key, val in FLAGS.__flags.items():  # for each flag
    if key in hparam_list:  # if it's in the list
      hps_dict[key] = val.value # add it to the dict
  hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

  # Create a batcher object that will create minibatches of data
  # batcher = Batcher(FLAGS.data_path, hps)

  tf.set_random_seed(111)  # a seed value for randomness
  print("creating model...")

  if FLAGS.dataset_id == 'mnist':
    if FLAGS.model == 'vanilla_dc':
      from model_DCGAN_vanilla import GAN_model
    elif FLAGS.model == 'vanilla':
      from model_vanilla_autoInput import GAN_model
    elif FLAGS.model == 'ToM':
      from model_ToM import GAN_model
    elif FLAGS.model == 'ToM_batch':
      from model_ToM_batch import GAN_model
    elif FLAGS.model == 'ToM_while':
      from model_ToM_while import GAN_model
    elif FLAGS.model == 'ToM_DC':
      from model_ToM_DC import GAN_model
    elif FLAGS.model == 'ToM_DC_batch':
      from model_ToM_DC_batch import GAN_model
    elif FLAGS.model == 'ToM_DC_while':
      from model_ToM_DC_while import GAN_model
    else:
      raise ValueError("Model name does not exist!")
  elif FLAGS.dataset_id == 'cifar':
    if FLAGS.model == 'vanilla_dc':
      from model_DCGAN_vanilla_cifar import GAN_model
    elif FLAGS.model == 'vanilla':
      from model_vanilla_autoInput import GAN_model
    elif FLAGS.model == 'ToM':
      from model_ToM import GAN_model
    elif FLAGS.model == 'ToM_batch':
      from model_ToM_batch import GAN_model
    elif FLAGS.model == 'ToM_while':
      from model_ToM_while import GAN_model
    elif FLAGS.model == 'ToM_DC':
      from model_ToM_DC import GAN_model
    elif FLAGS.model == 'ToM_DC_batch':
      from model_ToM_DC_batch import GAN_model
    elif FLAGS.model == 'ToM_DC_while':
      from model_ToM_DC_while import GAN_model
    else:
      raise ValueError("Model name does not exist!")

  model = GAN_model(hps)
  setup_training(model)


if __name__ == '__main__':
  tf.app.run()
