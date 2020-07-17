# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pre-trains an ELECTRA model."""

import argparse
import collections
import json
import time
import datetime
import os

import tensorflow as tf
import horovod.tensorflow as hvd
from horovod.tensorflow.compression import Compression

import utils
import pretrain_utils
from utils import get_rank, get_world_size, is_main_process, log, log_config
from configuration import ElectraConfig
from tokenization import ElectraTokenizer
from modeling import TFElectraForPreTraining, TFElectraForMaskedLM
from optimization import create_optimizer


class PretrainingConfig(object):
    """Defines pre-training hyperparameters."""

    def __init__(self, model_name, data_dir, **kwargs):
        self.model_name = model_name
        self.data_dir = data_dir

        self.debug = False  # debug mode for quickly running things
        self.do_train = True  # pre-train ELECTRA
        self.do_eval = False  # evaluate generator/discriminator on unlabeled data

        self.seed = 42

        # amp
        self.amp = True
        self.xla = True

        # optimizer type
        self.optimizer = 'adam'

        # loss functions
        self.electra_objective = True  # if False, use the BERT objective instead
        self.gen_weight = 1.0  # masked language modeling / generator loss
        self.disc_weight = 50.0  # discriminator loss
        self.mask_prob = 0.15  # percent of input tokens to mask out / replace

        # optimization
        self.learning_rate = 5e-4
        self.lr_decay_power = 1.0  # linear weight decay by default
        self.weight_decay_rate = 0.01
        self.num_warmup_steps = 10000

        # training settings
        self.save_checkpoints_steps = 1000
        self.num_train_steps = 1000000
        self.num_eval_steps = 100
        self.keep_checkpoint_max = 5  # maximum number of recent checkpoint files to keep;
        self.restore_checkpoint = False
        # change to 0 or None to keep all checkpoints

        # model settings
        self.model_size = "small"  # one of "small", "base", or "large"
        # override the default transformer hparams for the provided model size; see
        # modeling.BertConfig for the possible hparams and util.training_utils for
        # the defaults
        self.model_hparam_overrides = (
            kwargs["model_hparam_overrides"]
            if "model_hparam_overrides" in kwargs else {})
        self.embedding_size = None  # bert hidden size by default
        self.vocab_size = 30522  # number of tokens in the vocabulary
        self.do_lower_case = True  # lowercase the input?

        # generator settings
        self.uniform_generator = False  # generator is uniform at random
        self.shared_embeddings = True  # share generator/discriminator token embeddings?
        # self.untied_generator = True  # tie all generator/discriminator weights?
        self.generator_layers = 1.0  # frac of discriminator layers for generator
        self.generator_hidden_size = 0.25  # frac of discrim hidden size for gen
        self.disallow_correct = False  # force the generator to sample incorrect
        # tokens (so 15% of tokens are always
        # fake)
        self.temperature = 1.0  # temperature for sampling from generator

        # batch sizes
        self.max_seq_length = 128
        self.train_batch_size = 128
        self.eval_batch_size = 128

        # default locations of data files
        self.pretrain_tfrecords = os.path.join(
            data_dir, "pretrain_tfrecords/pretrain_data.tfrecord*")
        self.vocab_file = os.path.join(data_dir, "vocab.txt")
        self.model_dir = os.path.join(data_dir, "models", model_name)
        self.checkpoints_dir = os.path.join(self.model_dir, "checkpoints")
        results_dir = os.path.join(self.model_dir, "results")
        self.results_txt = os.path.join(results_dir, "unsup_results.txt")
        self.results_pkl = os.path.join(results_dir, "unsup_results.pkl")
        self.log_dir = os.path.join(self.model_dir, "logs")

        self.max_predictions_per_seq = int((self.mask_prob + 0.005) *
                                           self.max_seq_length)

        # debug-mode settings
        if self.debug:
            self.train_batch_size = 8
            self.num_train_steps = 20
            self.eval_batch_size = 4
            self.num_eval_steps = 2

        # defaults for different-sized model
        if self.model_size == "small":
            self.embedding_size = 128
        # Here are the hyperparameters we used for larger models; see Table 6 in the
        # paper for the full hyperparameters
        # else:
        #   self.max_seq_length = 512
        #   self.learning_rate = 2e-4
        #   if self.model_size == "base":
        #     self.embedding_size = 768
        #     self.generator_hidden_size = 0.33333
        #     self.train_batch_size = 256
        #   else:
        #     self.embedding_size = 1024
        #     self.mask_prob = 0.25
        #     self.train_batch_size = 2048

        # passed-in-arguments override defaults
        self.update(kwargs)

    def update(self, kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__:
                raise ValueError("Unknown hparam " + k)
            if v is not None:
                self.__dict__[k] = v


class PretrainingModel(tf.keras.Model):
    """Transformer pre-training using the replaced-token-detection task."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # Set up model config
        self._config = config
        self.disc_config = ElectraConfig(vocab_size=30522,
                                         embedding_size=768,
                                         hidden_size=768,
                                         num_hidden_layers=12,
                                         num_attention_heads=12,
                                         intermediate_size=3072,
                                         hidden_act="gelu",
                                         hidden_dropout_prob=0.1,
                                         attention_probs_dropout_prob=0.1,)
        if config.debug:
            self.disc_config.num_hidden_layers = 3
            self.disc_config.hidden_size = 128
            self.disc_config.intermediate_size = 128 * 4
            self.disc_config.num_attention_heads = 4

        # Set up discriminator
        self.discriminator = TFElectraForPreTraining(self.disc_config)

        # Set up generator
        gen_config = get_generator_config(config, self.disc_config)
        if config.electra_objective:
            if config.shared_embeddings:
                self.generator = TFElectraForMaskedLM(
                    gen_config, shared_embeddings=True,
                    input_embeddings=self.discriminator.get_input_embeddings())
            else:
                self.generator = TFElectraForMaskedLM(gen_config)
        else:
            self.generator = TFElectraForMaskedLM(self.disc_config)

    def call(self, features, is_training):
        config = self._config

        # Mask the input
        masked_inputs = pretrain_utils.mask(
            config, pretrain_utils.features_to_inputs(features), config.mask_prob)

        # Generator
        if config.uniform_generator:
            mlm_output = self._get_masked_lm_output(masked_inputs, None, is_training=is_training)
        else:
            mlm_output = self._get_masked_lm_output(
                masked_inputs, self.generator, is_training=is_training)
        fake_data = self._get_fake_data(masked_inputs, mlm_output.logits)
        total_loss = config.gen_weight * mlm_output.loss

        # Discriminator
        disc_output = None
        if config.electra_objective:
            disc_output = self._get_discriminator_output(
                fake_data.inputs, self.discriminator, fake_data.is_fake_tokens,
                is_training=is_training)
            total_loss += config.disc_weight * disc_output.loss

        # Evaluation inputs
        eval_fn_inputs = {
            "input_ids": masked_inputs.input_ids,
            "masked_lm_preds": mlm_output.preds,
            "mlm_loss": mlm_output.per_example_loss,
            "masked_lm_ids": masked_inputs.masked_lm_ids,
            "masked_lm_weights": masked_inputs.masked_lm_weights,
            "input_mask": masked_inputs.input_mask
        }
        if config.electra_objective:
            eval_fn_inputs.update({
                "disc_loss": disc_output.per_example_loss,
                "disc_labels": disc_output.labels,
                "disc_probs": disc_output.probs,
                "disc_preds": disc_output.preds,
                "sampled_tokids": tf.argmax(fake_data.sampled_tokens, -1,
                                            output_type=tf.int32)
            })

        return total_loss, eval_fn_inputs

    def _get_masked_lm_output(self, inputs, generator, is_training=False):
        """Masked language modeling softmax layer."""
        masked_lm_weights = inputs.masked_lm_weights

        if self._config.uniform_generator:
            logits = tf.zeros(self.disc_config.vocab_size)
            logits_tiled = tf.zeros(
                pretrain_utils.get_shape_list(inputs.masked_lm_ids) +
                [self.disc_config.vocab_size])
            logits_tiled += tf.reshape(logits, [1, 1, self.disc_config.vocab_size])
            logits = logits_tiled
        else:
            outputs = generator(
                input_ids=inputs.input_ids,
                attention_mask=inputs.input_mask,
                token_type_ids=inputs.segment_ids,
                training=is_training)
            logits = outputs[0]
            logits = pretrain_utils.gather_positions(
                logits, inputs.masked_lm_positions)

        oh_labels = tf.one_hot(
            inputs.masked_lm_ids, depth=self.disc_config.vocab_size,
            dtype=tf.float32)

        probs = tf.nn.softmax(logits)
        log_probs = tf.nn.log_softmax(logits)
        label_log_probs = -tf.reduce_sum(log_probs * oh_labels, axis=-1)

        numerator = tf.reduce_sum(inputs.masked_lm_weights * label_log_probs)
        denominator = tf.reduce_sum(masked_lm_weights) + 1e-6
        loss = numerator / denominator
        preds = tf.argmax(log_probs, axis=-1, output_type=tf.int32)

        MLMOutput = collections.namedtuple(
            "MLMOutput", ["logits", "probs", "loss", "per_example_loss", "preds"])
        return MLMOutput(
            logits=logits, probs=probs, per_example_loss=label_log_probs,
            loss=loss, preds=preds)

    def _get_discriminator_output(self, inputs, discriminator, labels, is_training=False):
        """Discriminator binary classifier."""

        outputs = discriminator(
            input_ids=inputs.input_ids,
            attention_mask=inputs.input_mask,
            token_type_ids=inputs.segment_ids,
            training=is_training,
        )
        logits = outputs[0]
        weights = tf.cast(inputs.input_mask, tf.float32)
        labelsf = tf.cast(labels, tf.float32)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labelsf) * weights
        per_example_loss = (tf.reduce_sum(losses, axis=-1) /
                            (1e-6 + tf.reduce_sum(weights, axis=-1)))
        loss = tf.reduce_sum(losses) / (1e-6 + tf.reduce_sum(weights))
        probs = tf.nn.sigmoid(logits)
        preds = tf.cast(tf.round((tf.sign(logits) + 1) / 2), tf.int32)
        DiscOutput = collections.namedtuple(
            "DiscOutput", ["loss", "per_example_loss", "probs", "preds",
                           "labels"])
        return DiscOutput(
            loss=loss, per_example_loss=per_example_loss, probs=probs,
            preds=preds, labels=labels,
        )

    def _get_fake_data(self, inputs, mlm_logits):
        """Sample from the generator to create corrupted input."""
        inputs = pretrain_utils.unmask(inputs)
        disallow = tf.one_hot(
            inputs.masked_lm_ids, depth=self.disc_config.vocab_size,
            dtype=tf.float32) if self._config.disallow_correct else None
        sampled_tokens = tf.stop_gradient(pretrain_utils.sample_from_softmax(
            mlm_logits / self._config.temperature, disallow=disallow))
        sampled_tokids = tf.argmax(sampled_tokens, -1, output_type=tf.int32)
        updated_input_ids, masked = pretrain_utils.scatter_update(
            inputs.input_ids, sampled_tokids, inputs.masked_lm_positions)
        labels = masked * (1 - tf.cast(
            tf.equal(updated_input_ids, inputs.input_ids), tf.int32))
        updated_inputs = pretrain_utils.get_updated_inputs(
            inputs, input_ids=updated_input_ids)
        FakedData = collections.namedtuple("FakedData", [
            "inputs", "is_fake_tokens", "sampled_tokens"])
        return FakedData(inputs=updated_inputs, is_fake_tokens=labels,
                         sampled_tokens=sampled_tokens)


def metric_fn(config, metrics, eval_fn_inputs):
    """Computes the loss and accuracy of the model."""
    d = eval_fn_inputs
    metrics["masked_lm_accuracy"].update_state(
        y_true=tf.reshape(d["masked_lm_ids"], [-1]),
        y_pred=tf.reshape(d["masked_lm_preds"], [-1]),
        sample_weight=tf.reshape(d["masked_lm_weights"], [-1]))
    metrics["masked_lm_loss"].update_state(
        values=tf.reshape(d["mlm_loss"], [-1]),
        sample_weight=tf.reshape(d["masked_lm_weights"], [-1]))
    if config.electra_objective:
        metrics["sampled_masked_lm_accuracy"].update_state(
            y_true=tf.reshape(d["masked_lm_ids"], [-1]),
            y_pred=tf.reshape(d["sampled_tokids"], [-1]),
            sample_weight=tf.reshape(d["masked_lm_weights"], [-1]))
        if config.disc_weight > 0:
            metrics["disc_loss"].update_state(d["disc_loss"])
            metrics["disc_auc"].update_state(
                d["disc_labels"] * d["input_mask"],
                d["disc_probs"] * tf.cast(d["input_mask"], tf.float32))
            metrics["disc_accuracy"].update_state(
                y_true=d["disc_labels"], y_pred=d["disc_preds"],
                sample_weight=d["input_mask"])
            metrics["disc_precision"].update_state(
                y_true=d["disc_labels"], y_pred=d["disc_preds"],
                sample_weight=d["disc_preds"] * d["input_mask"])
            metrics["disc_recall"].update_state(
                y_true=d["disc_labels"], y_pred=d["disc_preds"],
                sample_weight=d["disc_labels"] * d["input_mask"])
    return metrics


def get_generator_config(config, bert_config):
    """Get model config for the generator network."""
    gen_config = ElectraConfig.from_dict(bert_config.to_dict())
    gen_config.hidden_size = int(round(
        bert_config.hidden_size * config.generator_hidden_size))
    gen_config.num_hidden_layers = int(round(
        bert_config.num_hidden_layers * config.generator_layers))
    gen_config.intermediate_size = 4 * gen_config.hidden_size
    gen_config.num_attention_heads = max(1, gen_config.hidden_size // 64)
    return gen_config


@tf.function
def train_one_step(config, model, optimizer, features, init=False, clip_norm=1.0):
    with tf.GradientTape() as tape:
        total_loss, eval_fn_inputs = model(features, is_training=True)
        unscaled_loss = tf.stop_gradient(total_loss)
        if config.amp:
            total_loss = optimizer.get_scaled_loss(total_loss)

    tape = hvd.DistributedGradientTape(
        tape, sparse_as_dense=True)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    if config.amp:
        gradients = optimizer.get_unscaled_gradients(gradients)
    (gradients, _) = tf.clip_by_global_norm(gradients, clip_norm=clip_norm)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if init:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(optimizer.variables(), root_rank=0)
    return unscaled_loss, eval_fn_inputs


def main():
    # Parse essential args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Location of data files (model weights, etc).")
    parser.add_argument("--model_name", required=True,
                        help="The name of the model being fine-tuned.")
    parser.add_argument("--pretrain_tfrecords", type=str)

    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_train_steps", type=int)
    parser.add_argument("--num_warmup_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--max_seq_length", type=int)

    parser.add_argument("--mask_prob", type=float)
    parser.add_argument("--disc_weight", type=float)
    parser.add_argument("--generator_hidden_size", type=float)

    parser.add_argument("--save_checkpoints_steps", type=int)
    parser.add_argument("--keep_checkpoint_max", type=int)
    parser.add_argument("--restore_checkpoint", action='store_true')

    parser.add_argument("--optimizer", default="adam", type=str, help="adam or lamb")

    args = parser.parse_args()
    config = PretrainingConfig(**args.__dict__)

    # Set up tensorflow
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    tf.config.optimizer.set_jit(config.xla)
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": config.amp})
    tf.random.set_seed(config.seed)

    # Set up config
    if config.do_train == config.do_eval:
        raise ValueError("Exactly one of `do_train` or `do_eval` must be True.")
    if config.debug and config.do_train:
        utils.rmkdir(config.model_dir)
    utils.heading("Config:")
    log_config(config)

    # Save pretrain configs
    pretrain_config_json = os.path.join(config.checkpoints_dir, 'pretrain_config.json')
    if is_main_process():
        utils.write_json(config.__dict__, pretrain_config_json)
        log("Configuration saved in {}".format(pretrain_config_json))

    # Set up model
    model = PretrainingModel(config)

    # Set up metrics
    perf_metrics = dict()
    perf_metrics["train_perf"] = tf.keras.metrics.Mean(name="train_perf")

    eval_metrics = dict()
    eval_metrics["total_loss"] = tf.keras.metrics.Mean(name="total_loss")
    eval_metrics["masked_lm_accuracy"] = tf.keras.metrics.Accuracy(name="masked_lm_accuracy")
    eval_metrics["masked_lm_loss"] = tf.keras.metrics.Mean(name="masked_lm_loss")
    if config.electra_objective:
        eval_metrics["sampled_masked_lm_accuracy"] = tf.keras.metrics.Accuracy(name="sampled_masked_lm_accuracy")
        if config.disc_weight > 0:
            eval_metrics["disc_loss"] = tf.keras.metrics.Mean(name="disc_loss")
            eval_metrics["disc_auc"] = tf.keras.metrics.AUC(name="disc_auc")
            eval_metrics["disc_accuracy"] = tf.keras.metrics.Accuracy(name="disc_accuracy")
            eval_metrics["disc_precision"] = tf.keras.metrics.Accuracy(name="disc_precision")
            eval_metrics["disc_recall"] = tf.keras.metrics.Accuracy(name="disc_recall")

    # Set up tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(config.log_dir, current_time,
                                 'train_' + str(get_rank()) + '_of_' + str(get_world_size()))
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Set up dataset
    dataset = pretrain_utils.get_dataset(
        config, config.train_batch_size, world_size=get_world_size(), rank=get_rank())
    train_iterator = iter(dataset)

    # Set up optimizer
    optimizer = create_optimizer(
        init_lr=config.learning_rate,
        num_train_steps=config.num_train_steps,
        num_warmup_steps=config.num_warmup_steps,
        weight_decay_rate=config.weight_decay_rate,
        optimizer=config.optimizer)
    if config.amp:
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")

    if config.do_train:
        # Set up checkpoint manager
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(checkpoint, config.checkpoints_dir, max_to_keep=config.keep_checkpoint_max)
        iter_checkpoint = tf.train.Checkpoint(train_iterator=train_iterator)
        iter_manager = tf.train.CheckpointManager(
            iter_checkpoint,
            os.path.join(config.checkpoints_dir, 'iter_ckpt_rank_' + '{:02}'.format(get_rank())),
            checkpoint_name='iter_ckpt_rank_' + '{:02}'.format(get_rank()),
            max_to_keep=config.keep_checkpoint_max)
        if config.restore_checkpoint and manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint)
            log(" ** Restored model checkpoint from {}".format(manager.latest_checkpoint))
            if iter_manager.latest_checkpoint:
                iter_checkpoint.restore(iter_manager.latest_checkpoint)
                log(" ** Restored iterator checkpoint from {}".format(iter_manager.latest_checkpoint), all_rank=True)
        else:
            log(" ** Initializing from scratch.")

        utils.heading("Running training")
        train_start, start_step = time.time(), int(checkpoint.step)-1
        while int(checkpoint.step) <= config.num_train_steps:
            step = int(checkpoint.step)
            features = next(train_iterator)
            iter_start = time.time()

            # if step == 200: tf.profiler.experimental.start(logdir=train_log_dir)
            total_loss, eval_fn_inputs = train_one_step(config, model, optimizer, features, step <= 1)
            # if step == 300: tf.profiler.experimental.stop()

            perf_metrics["train_perf"].update_state(
                config.train_batch_size * get_world_size() / (time.time() - iter_start))
            eval_metrics["total_loss"].update_state(values=total_loss)
            metric_fn(config, eval_metrics, eval_fn_inputs)

            if step % 100 == 0:
                log('Step:{:6d}, Loss:{:10.6f}, Gen_loss:{:10.6f}, Disc_loss:{:10.6f}, Gen_acc:{:6.2f}, '
                          'Disc_acc:{:6.2f}, Perf:{:4.0f}, Elapsed: {}, ETA: {}, '.format(
                    step, total_loss,
                    eval_metrics["masked_lm_loss"].result().numpy(),
                    eval_metrics["disc_loss"].result().numpy(),
                    eval_metrics["masked_lm_accuracy"].result().numpy() * 100,
                    eval_metrics["disc_accuracy"].result().numpy() * 100,
                    perf_metrics["train_perf"].result().numpy(),
                    utils.get_readable_time(time.time() - train_start),
                    utils.get_readable_time((time.time()-train_start)/(step-start_step)*(config.num_train_steps-step))),
                    all_rank=True)

                with train_summary_writer.as_default():
                    for key, m in eval_metrics.items():
                        tf.summary.scalar(key, m.result(), step=step)

                for m in eval_metrics.values():
                    m.reset_states()

            checkpoint.step.assign_add(1)
            if step % config.save_checkpoints_steps == 0:
                if is_main_process():
                    save_path = manager.save()
                    log(" ** Saved model checkpoint for step {}: {}".format(step, save_path))
                iter_save_path = iter_manager.save()
                log(" ** Saved iterator checkpoint for step {}: {}".format(step, iter_save_path), all_rank=True)

    if config.do_eval:
        pass

if __name__ == "__main__":
    main()
