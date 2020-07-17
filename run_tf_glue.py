import os
import time
import tensorflow as tf
import tensorflow_datasets
import horovod.tensorflow as hvd

# src/transformers/modeling_tf_electra.py:        from transformers import ElectraTokenizer, TFElectraModel
# src/transformers/modeling_tf_electra.py:        from transformers import ElectraTokenizer, TFElectraForPreTraining
# src/transformers/modeling_tf_electra.py:        from transformers import ElectraTokenizer, TFElectraForMaskedLM
# src/transformers/modeling_tf_electra.py:        from transformers import ElectraTokenizer, TFElectraForTokenClassification

from transformers import (
    BertConfig,
    # BertForSequenceClassification,
    BertTokenizer,
    TFBertForSequenceClassification,
    glue_convert_examples_to_features,
    glue_processors,
)

from transformers import (
    ElectraConfig,
    ElectraTokenizer,
    # TFElectraForTokenClassification,
    TFElectraPreTrainedModel,
)
from transformers.modeling_tf_electra import TFElectraMainLayer
import transformers.optimization_tf
from transformers.optimization_tf import create_optimizer
# from transformers import AdamWeightDecay


from transformers.data.processors import glue


class TFElectraForTokenClassification(TFElectraPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.electra = TFElectraMainLayer(config, name="electra")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(config.num_labels, name="classifier")

    def call(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            training=False,
    ):
        r"""
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.ElectraConfig`) and inputs:
        scores (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        import tensorflow as tf
        from transformers import ElectraTokenizer, TFElectraForTokenClassification
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        model = TFElectraForTokenClassification.from_pretrained('google/electra-small-discriminator')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        scores = outputs[0]
        """
        # print("HEloooooo inside calll \n\n\n==============")
        # print("inputs inside", input_ids)
        discriminator_hidden_states = self.electra(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, training=training
        )
        # print("did call electra")
        # print("hidden states", len(discriminator_hidden_states))
        discriminator_sequence_output = discriminator_hidden_states[0][:, 0]
        discriminator_sequence_output = self.dropout(discriminator_sequence_output)
        # print("seq output", discriminator_sequence_output.shape)
        logits = self.classifier(discriminator_sequence_output)
        output = (logits,)
        output += discriminator_hidden_states[1:]
        # print("logits", logits.shape)
        # print("retirn out", len(output))
        return output  # (loss), scores, (hidden_states), (attentions)


# from transformers.modeling_bert import BertForSequenceClassification

def get_dataset_from_features(features, batch_size, drop_remainder=False):
    """Input function for training"""

    # input_ids: List[int]
    # attention_mask: Optional[List[int]] = None
    # token_type_ids: Optional[List[int]] = None
    # label: Optional[Union[int, float]] = None

    all_input_ids = tf.convert_to_tensor([f.input_ids for f in features], dtype=tf.int64)
    all_input_mask = tf.convert_to_tensor([f.attention_mask for f in features], dtype=tf.int64)
    all_segment_ids = tf.convert_to_tensor([f.token_type_ids for f in features], dtype=tf.int64)
    all_label_ids = tf.convert_to_tensor([f.label for f in features], dtype=tf.int64)

    dataset = tf.data.Dataset.from_tensor_slices(
        (all_input_ids, all_input_mask, all_segment_ids, all_label_ids))
    dataset = dataset.shard(hvd.size(), hvd.rank())
    # dataset = dataset.repeat()
    dataset = dataset.shuffle(batch_size * 3)
    # dataset = dataset.map(self._preproc_samples,
    #                      num_parallel_calls=multiprocessing.cpu_count()//self._num_gpus)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(batch_size)

    return dataset


@tf.function
def train_step(model, inputs, loss, amp, opt, init):
    with tf.GradientTape() as tape:
        [input_ids, input_mask, segment_ids, label_ids] = inputs
        # print(input_ids, input_ids.shape)
        outputs = model(input_ids,
                        # input_ids=input_ids,
                        attention_mask=input_mask,
                        token_type_ids=segment_ids,
                        position_ids=None,
                        head_mask=None,
                        inputs_embeds=None,
                        training=True,
                        )

        loss_value = loss(y_true=label_ids, y_pred=outputs[0])
        unscaled_loss = tf.stop_gradient(loss_value)
        if amp:
            loss_value = opt.get_scaled_loss(loss_value)
    tape = hvd.DistributedGradientTape(tape)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    if amp:
        gradients = opt.get_unscaled_gradients(gradients)
    opt.apply_gradients(zip(gradients, model.trainable_variables))  # , clip_norm=1.0)

    if init:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)

    return unscaled_loss, outputs  # , tape.gradient(loss_value, model.trainable_variables)


# script parameters
BATCH_SIZE = 32
EVAL_BATCH_SIZE = BATCH_SIZE
USE_XLA = False
USE_AMP = True
EPOCHS = 3

TASK = "mrpc"
val_string = "validation"
test_string = "test"
if TASK == "sst-2":
    TFDS_TASK = "sst2"
elif TASK == "sts-b":
    TFDS_TASK = "stsb"
else:
    TFDS_TASK = TASK

if TASK == "mnli":
    val_string += "_matched"
    test_string += "_matched"

num_labels = len(glue.glue_processors[TASK]().get_labels())
print(num_labels)

hvd.init()
if USE_XLA:
    tf.config.optimizer.set_jit(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# tf.config.optimizer.set_jit(USE_XLA)
if USE_AMP:  # params.use_amp:
    # tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    tf.keras.mixed_precision.experimental.set_policy(policy)

# tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

# Load tokenizer and model from pretrained model/vocabulary. Specify the number of labels to classify (2+: classification, 1: regression)
config = ElectraConfig.from_pretrained("google/electra-base-discriminator", num_labels=num_labels)
tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
model = TFElectraForTokenClassification.from_pretrained("google/electra-base-discriminator", config=config)

# Load dataset via TensorFlow Datasets
# data, info = tensorflow_datasets.load(f"glue/{TFDS_TASK}", with_info=True)
# train_examples = info.splits["train"].num_examples

# MNLI expects either validation_matched or validation_mismatched
# valid_examples = info.splits[val_string].num_examples
# test_examples = info.splits[test_string].num_examples

##replace train and test examples
data_processor = glue.glue_processors[TASK]()
data_dir = os.environ['GLUE_DIR']
train_examples = data_processor.get_train_examples(data_dir)
dev_examples = data_processor.get_dev_examples(data_dir)
# Prepare dataset for GLUE as a tf.data.Dataset instance
# train_dataset = glue_convert_examples_to_features(data["train"], tokenizer, max_length=128, task=TASK)
train_features = glue.glue_convert_examples_to_features(train_examples, tokenizer, max_length=128, task=TASK)

# MNLI expects either validation_matched or validation_mismatched
# test_features = glue.glue_convert_examples_to_features(data[test_string], tokenizer, max_length=128, task=TASK)

# test_dataset = glue_convert_examples_to_features(data[test_string], tokenizer, max_length=128, task=TASK)
dev_features = glue.glue_convert_examples_to_features(dev_examples, tokenizer, max_length=128, task=TASK)
# dev_dataset_for_eval = glue.glue_convert_examples_to_features(data[val_string], tokenizer, max_length=128, task=TASK)
# old code
# train_dataset = train_dataset.shuffle(128).batch(BATCH_SIZE).repeat(-1)
# valid_dataset = valid_dataset.batch(EVAL_BATCH_SIZE)
# test_dataset = test_dataset.batch(EVAL_BATCH_SIZE)

# train_steps = train_examples // BATCH_SIZE
# valid_steps = valid_examples // EVAL_BATCH_SIZE
# test_steps = test_examples // EVAL_BATCH_SIZE
len_train_features = len(train_features)
len_dev_features = len(dev_features)
total_train_steps = int(len_train_features * EPOCHS / BATCH_SIZE) + 1
train_dataset = get_dataset_from_features(train_features, BATCH_SIZE)
dev_dataset = get_dataset_from_features(dev_features, EVAL_BATCH_SIZE)

# Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
opt = create_optimizer(init_lr=3e-5, num_train_steps=total_train_steps, num_warmup_steps=int(0.1 * total_train_steps))
# opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
if USE_AMP:
    # loss scaling is currently required when using mixed precision
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")

if num_labels == 1:
    loss = tf.keras.losses.MeanSquaredError()
else:
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
model.compile(optimizer=opt, loss=loss, metrics=[metric])
train_loss_results, train_accuracy_results = [], []
for epoch in range(EPOCHS):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    epoch_perf_avg = tf.keras.metrics.Mean()

    # Training loop - using batches of 32
    for iter, inputs in enumerate(train_dataset):
        iter_start = time.time()
        # Optimize the model
        loss_value, outputs = train_step(model, inputs, loss, USE_AMP, opt, (iter == 0 and epoch == 0))
        epoch_perf_avg.update_state(1. * len(inputs[0]) / (time.time() - iter_start))
        if iter % 10 == 0:
            print("step:{} loss:{}".format(iter, loss_value))
        # opt.apply_gradients(zip(grads, model.trainable_variables), clip_norm=1.0)

        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        epoch_accuracy.update_state(inputs[-1], outputs)

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
    if epoch % 1 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, Perf: {:.3f} seq/s".format(epoch,
                                                                                        epoch_loss_avg.result(),
                                                                                        epoch_accuracy.result(),
                                                                                        epoch_perf_avg.result(),
                                                                                        ))
##OLD code
# model.compile(optimizer=opt, loss=loss, metrics=[metric])
if hvd.rank() == 0:
    test_accuracy = tf.keras.metrics.Accuracy()
    infer_perf_avg = tf.keras.metrics.Mean()
    for input_ids, input_mask, segment_ids, label_ids in dev_dataset:
        # training=False is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        iter_start = time.time()
        logits = model(input_ids,
                       # input_ids=input_ids,
                       attention_mask=input_mask,
                       token_type_ids=segment_ids,
                       position_ids=None,
                       head_mask=None,
                       inputs_embeds=None,
                       training=False,
                       )[0]
        infer_perf_avg.update_state(1. * len(logits) / (time.time() - iter_start))
        predictions = tf.argmax(logits, axis=1)
        test_accuracy(predictions, label_ids)

    print("Test set accuracy: {:.3%}, perf: {:.3f} seq/s".format(test_accuracy.result(), infer_perf_avg.result()))

## Train and evaluate using tf.keras.Model.fit()
# history = model.fit(
#    train_dataset,
#    epochs=EPOCHS,
#    steps_per_epoch=train_steps,
#    validation_data=valid_dataset,
#    validation_steps=valid_steps,
# )
print("evaluate")
# model.evaluate(dev_dataset_for_eval)
# Save TF2 model
os.makedirs("./save/", exist_ok=True)
if hvd.rank() == 0:
    model.save_pretrained("./save/")

if False:  # TASK == "mrpc":
    # Load the TensorFlow model in PyTorch for inspection
    # This is to demo the interoperability between the two frameworks, you don't have to
    # do this in real life (you can run the inference on the TF model).
    pytorch_model = model  # BertForSequenceClassification.from_pretrained("./save/", from_tf=True)

    # Quickly test a few predictions - MRPC is a paraphrasing task, let's see if our model learned the task
    sentence_0 = "This research was consistent with his findings."
    sentence_1 = "His findings were compatible with this research."
    sentence_2 = "His findings were not compatible with this research."
    inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=True, return_tensors="pt")
    inputs_2 = tokenizer.encode_plus(sentence_0, sentence_2, add_special_tokens=True, return_tensors="pt")

    pred_1 = pytorch_model(**inputs_1)[0].argmax().item()
    pred_2 = pytorch_model(**inputs_2)[0].argmax().item()
    print("sentence_1 is", "a paraphrase" if pred_1 else "not a paraphrase", "of sentence_0")
    print("sentence_2 is", "a paraphrase" if pred_2 else "not a paraphrase", "of sentence_0")
