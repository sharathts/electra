import argparse
import collections
import json
import os

import tensorflow as tf

import utils
from run_pretrain import PretrainingConfig, PretrainingModel


def from_pretrained_ckpt(args):
    config = PretrainingConfig(
        model_name='postprocessing',
        data_dir='postprocessing',
        generator_hidden_size=0.3333333,
        max_seq_length=512,
    )

    # Set up model
    model = PretrainingModel(config)

    # Load checkpoint
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), model=model)
    checkpoint.restore(args.pretrained_checkpoint).expect_partial()
    utils.log(" ** Restored from {} at step {}".format(args.pretrained_checkpoint, int(checkpoint.step)-1))

    model.discriminator(model.discriminator.dummy_inputs)
    model.discriminator.save_pretrained(args.output_dir)


if __name__ == '__main__':
    # Parse essential args
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_checkpoint')
    parser.add_argument('--pretrain_config')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    from_pretrained_ckpt(args)
