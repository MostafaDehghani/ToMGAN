import os
import time
from collections import namedtuple

import numpy as np
import tensorflow as tf
from batcher import Batcher
from model_0 import TomGAN_model
from tensorflow.python import debug as tf_debug

import util

FLAGS = tf.app.flags.FLAGS

# Where to save output
tf.app.flags.DEFINE_string('log_root', '../log_root', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('data_path', '../data', 'Directory where the data '
                                            'is going to be saved.')
tf.app.flags.DEFINE_string('exp_name', 'main_experiment', 'Name for experiment. Logs will '
                                           'be saved in a directory with this'
                                           ' name, under log_root.')
tf.app.flags.DEFINE_string('model', 'vanilla', 'must be one of '
                                               'vanila/tomI/tomII')
tf.app.flags.DEFINE_integer('hidden_dim', 128, 'dimension of hidden states '
                                               'in discriminator and generator')
tf.app.flags.DEFINE_integer('batch_size', 128, 'minibatch size')

tf.app.flags.DEFINE_integer('dis_output_size', 1, 'size of the input for the '
                                         'discriminator (1)')

tf.app.flags.DEFINE_integer('dis_input_size', 784, 'size of the input for the '
                                         'discriminator (for mnist = 784)')

tf.app.flags.DEFINE_integer('gen_input_size', 100, 'size of the noise vector '
                                          'as the input for the generator')
tf.app.flags.DEFINE_integer('gen_output_size', 784, 'size of the generator output vector')
tf.app.flags.DEFINE_integer('logging_step', 1000, 'logging step')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.app.flags.DEFINE_boolean('debug', False,
                "Run in tensorflow's debug mode (watches for NaN/inf values)")

# def restore_last_model():
#   exit()

def setup_training(model, batcher):
  """Does setup before starting training (run_training)"""
  train_dir = os.path.join(FLAGS.log_root, "train")
  if not os.path.exists(train_dir): os.makedirs(train_dir)

  model.build_graph() # build the graph
  # if FLAGS.restore_last_model:
  #   restore_last_model()
  saver = tf.train.Saver(max_to_keep=3) # keep 3 checkpoints at a time

  sv = tf.train.Supervisor(logdir=train_dir,
                     is_chief=True,
                     saver=saver,
                     summary_op=None,
                     save_summaries_secs=60, # save summaries for tensorboard every 60 secs
                     save_model_secs=60, # checkpoint every 60 secs
                     global_step=model.global_step_G
                     )

  summary_writer = sv.summary_writer
  tf.logging.info("Preparing or waiting for session...")
  sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
  tf.logging.info("Created session.")
  try:
    run_training(model, batcher, sess_context_manager, summary_writer)
    # this is an infinite loop until interrupted
  except KeyboardInterrupt:
    tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
    sv.stop()


def run_training(model, batcher, sess_context_manager, summary_writer):
  """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
  tf.logging.info("starting run_training")
  with sess_context_manager as sess:
    if FLAGS.debug: # start the tensorflow debugger
      sess = tf_debug.LocalCLIDebugWrapperSession(sess)
      sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    train_step = 0
    while True: # repeats until interrupted
      batch = batcher.next_batch()
      if train_step % FLAGS.logging_step == 0:
        tf.logging.info('------ training step: ' + str(train_step) + ' ------')
        t0=time.time()
        results = model.run_train_step(sess, batch)
        t1=time.time()
        tf.logging.info('seconds for training step: %.3f', t1-t0)

        loss_D = results['loss_D']
        tf.logging.info('loss_D: %f', loss_D)  # print the loss to screen

        loss_G = results['loss_G']
        tf.logging.info('loss_G: %f', loss_G) # print the loss to screen

        if not np.isfinite(loss_G):
          raise Exception("Loss_G is not finite. Stopping.")
        if not np.isfinite(loss_D):
          raise Exception("Loss_D is not finite. Stopping.")

        # flush the summary writer every so often
        summary_writer.flush()

        tf.logging.info("sampling from the generator")
        sampling_result  = model.sample_generator(sess)
        util.plot(sampling_result['g_sample'], train_step)

      else:  # no logging
        results = model.run_train_step(sess, batch)

      # we will write these summaries to tensorboard using summary_writer
      summaries = results['summaries']
      train_step_G = results['global_step_G']
      train_step_D = results['global_step_D']
      summary_writer.add_summary(summaries, train_step_G)  # write the summaries
      summary_writer.add_summary(summaries, train_step_D)  # write the summaries

      train_step += 1




def main(unused_argv):
  # if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
  #   raise Exception("Problem with flags: %s" % unused_argv)

  tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want

  # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
  FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
  if not os.path.exists(FLAGS.log_root):
    os.makedirs(FLAGS.log_root)

  # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
  hparam_list = ['batch_size', 'hidden_dim', 'dis_input_size', 'gen_input_size', 'dis_output_size', 'gen_output_size' ]
  hps_dict = {}
  for key,val in FLAGS.__flags.items(): # for each flag
    if key in hparam_list: # if it's in the list
      hps_dict[key] = val # add it to the dict
  hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

  # Create a batcher object that will create minibatches of data
  batcher = Batcher(FLAGS.data_path, hps)

  tf.set_random_seed(111) # a seed value for randomness
  print("creating model...")
  model = TomGAN_model(hps)
  setup_training(model, batcher)

if __name__ == '__main__':
  tf.app.run()