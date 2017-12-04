
"""This file contains some utility functions"""

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
FLAGS = tf.app.flags.FLAGS

def get_config():
  """Returns config for tf.session"""
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  return config

def load_ckpt(saver, sess, ckpt_dir="train"):
  """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
  while True:
    try:
      ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
      ckpt_state = tf.train.get_checkpoint_state(ckpt_dir,
                                                 latest_filename=None)
      tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
      saver.restore(sess, ckpt_state.model_checkpoint_path)
      return ckpt_state.model_checkpoint_path
    except:
      tf.logging.info("Failed to load checkpoint from %s. "
                      "Sleeping for %i secs...", ckpt_dir, 10)

def plot(samples, train_step):
  fig = plt.figure(figsize=(4, 4))
  gs = gridspec.GridSpec(4, 4)
  gs.update(wspace=0.05, hspace=0.05)

  for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

  plots_path = os.path.join(FLAGS.log_root, FLAGS.exp_name, 'gen_samples')
  if not os.path.exists(plots_path): os.makedirs(plots_path)

  plt.savefig(plots_path + '/{}.png'.format(str(train_step).zfill(3)),
              bbox_inches='tight')

  tf.logging.info("sample from generator has been saved: " +
                  plots_path + '/{}.png'.format(str(train_step).zfill(3)))

  plt.close(fig)

  return fig