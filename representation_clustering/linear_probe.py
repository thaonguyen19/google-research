import tensorflow as tf
import tensorflow_datasets as tfds
import flax
import functools
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os
import fs as pyfs
import pickle
from typing import Any, Dict, Union
from compute_purity_ami import load_eval_ds, get_learning_rate
from train import create_train_state
from configs.default_breeds import get_config
from flax.training import checkpoints
import linear_eval

BREEDS_INFO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "breeds")


@functools.partial(jax.jit, backend='cpu')
def compute_gram(x):
  with jax.experimental.enable_x64():
    x = x.astype(jnp.float64)
    gram = jnp.matmul(x, x.T)
    return gram


@functools.partial(jax.jit, backend='cpu')
def compute_v(x, u, s_squared):
  with jax.experimental.enable_x64():
    thr = 1e-15 * np.max(s_squared)
    s_inv = jnp.where(s_squared > thr, 1 / jnp.sqrt(s_squared),
                      jnp.float64(0.0))
    vt = jnp.matmul((u * s_inv[None, :]).T, x)
    return vt.astype(np.float32).T


jax_matmul = jax.jit(jnp.matmul, backend='cpu')


def efficient_tune_hparams_and_compute_test_accuracy(
    embeddings_train: jnp.ndarray, labels_train: jnp.ndarray,
    embeddings_val: jnp.ndarray, labels_val: jnp.ndarray,
    embeddings_test: jnp.ndarray, labels_test: jnp.ndarray,
    perform_rotation: Union[bool, None] = None,
    **kwargs):
  """Efficient transfer learning evaluation.
  
  To improve efficiency, when the embedding dimension is high, uses
  SVD to find a rotation that reduces dimensionality. The solution of
  logistic regression is equivariant to this rotation and the accuracy is
  invariant (up to numerical precision).

  Args:
    train_embeddings: `n x p` matrix of train embeddings.
    train_labels: Vector of length `n` containing integer labels for train set.
    val_embeddings: `n x p` matrix of train embeddings.
    val_labels: Vector of length `n` containing integer labels for val set.
    test_embeddings: `n x p` matrix of test embeddings.
    test_labels: Vector of length `n` containing integer labels for test set.
    perform_rotation: Whether to rotate embedding space. If `None`, pick an
      option that is likely to be efficient based on the number of examples
      and embedding dimension.
    kwargs: Additional kwargs to be passed to
      `linear_eval.tune_hparams_and_compute_test_accuracy`.
  
  Returns:
    A Dict containing:
      test_accuracy: Accuracy on test set.
      val_accuracy: Accuracy of best model on val set.
      optimal_l2_reg: Optimal value of L2 regularization.
      weights: Linear classifier weights.
      biases: Linear classifier biases.
  """
  if perform_rotation is None:
    perform_rotation = (
        embeddings_train.shape[1] > 20 * embeddings_train.shape[0])

  if perform_rotation:
    # Rotate the embeddings so that their dimensionality is reduced.
    x = np.concatenate((embeddings_train, embeddings_val))
    s_squared, u = np.linalg.eigh(compute_gram(x))
    thr = 1e-15 * np.max(s_squared)
    s_inv = np.zeros(s_squared.shape)
    s_inv[s_squared > thr] = 1 / np.sqrt(s_squared[s_squared > thr])
    with jax.experimental.enable_x64():
      v = jax_matmul((u * s_inv[None, :]).T, x).astype(np.float32).T
    del x
    embeddings_train = jax_matmul(embeddings_train, v)
    embeddings_val = jax_matmul(embeddings_val, v)
    embeddings_test = jax_matmul(embeddings_test, v)

  with jax.default_matmul_precision('float32'):
    out = linear_eval.tune_hparams_and_compute_test_accuracy(
      embeddings_train, labels_train,
      embeddings_val, labels_val,
      embeddings_test, labels_test)

  if perform_rotation:
    out['weights'] = jax_matmul(v, out['weights'])

  return out


#@flax.jax_utils.pad_shard_unpad
#@jax.pmap
def embed_images(model, state, images, layer_name):
  variables = {
      "params": state.params if state.ema_params is None else state.ema_params,
      "batch_stats": state.batch_stats
  }
  _, state = model.apply(variables, images, capture_intermediates=True, mutable=["intermediates"], train=False)
  if 'stage' in layer_name:
    stage, block = layer_name.split('_')
    out = state['intermediates'][stage][block]['__call__'][0]
  else:
    out = state['intermediates'][layer_name]['__call__'][0]
  return np.array(jnp.reshape(out, (out.shape[0], -1)))


def embed_dataset(model, state, ds, layer_name):
  all_embeddings = []
  all_labels = []
  for batch in ds:
    all_embeddings.append(embed_images(model, state, batch['image'], layer_name))
    all_labels.append(batch['label'])
  all_embeddings = np.concatenate(all_embeddings, 0)
  all_labels = np.concatenate(all_labels, 0)
  return all_embeddings, all_labels


if __name__ == '__main__':
  dataset_type = "entity13_4_subclasses"
  if '4_subclasses' in dataset_type:
    num_subclasses = 4
  else:
    num_subclasses = -1
  shuffle_subclasses = False
  model_dir = f"gs://representation_clustering/{dataset_type}_vgg16/"
  #model_dir = f"gs://gresearch/representation-interpretability/breeds/{dataset_type}_400_epochs_ema_0.99_bn_0.99/"
  ckpt_number = 81
  num_classes = 13
  if "shuffle" in model_dir:
    assert(shuffle_subclasses == True)

  dataset_type = dataset_type.split('_')[0] 
  config = get_config()
  learning_rate_fn = functools.partial(
      get_learning_rate,
      base_learning_rate=0.1,
      steps_per_epoch=40,
      num_epochs=config.num_epochs,
      warmup_epochs=config.warmup_epochs)
  checkpoint_path = os.path.join(model_dir, f'checkpoints-0/ckpt-{ckpt_number}.flax')
  model, state = create_train_state(config, jax.random.PRNGKey(0), input_shape=(8, 224, 224, 3), num_classes=num_classes, learning_rate_fn=learning_rate_fn)
  state = checkpoints.restore_checkpoint(checkpoint_path, state)
  print("step:", state.step)

  if 'vgg' in model_dir:
    layer_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 
                        'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
  else:
    layer_names = ['stage1_block1', 'stage1_block2', 'stage1_block3', 'stage2_block1', 'stage2_block2', \
                   'stage2_block3', 'stage2_block4', 'stage3_block1', 'stage3_block2', 'stage3_block3', \
                   'stage3_block4', 'stage3_block5', 'stage3_block6', 'stage4_block1', 'stage4_block2', 'stage4_block3']

  embeddings = {}
  labels = {}
  for layer_name in layer_names:
    if 'fc' in layer_name or 'conv1' in layer_name:
      continue
    print(f"################## LAYER = {layer_name}")
    for split_name, split in [('train', 'train[50000:]'),
                              ('val', 'train[:50000]'),
                              ('test', 'validation')]:
      ds, num_classes, _ = load_eval_ds(dataset_type, -1, num_subclasses, shuffle_subclasses, split=split, lookup_labels=True, use_fine_grained_labels=True)
      num_examples = int(ds.reduce(0, lambda x, _: x + 1).numpy())
      print(split_name, num_examples)
      embeddings[split_name], labels[split_name] = embed_dataset(model, state, ds, layer_name)

    out = efficient_tune_hparams_and_compute_test_accuracy(
      embeddings['train'], labels['train'],
      embeddings['val'], labels['val'],
      embeddings['test'], labels['test'],
      perform_rotation=False,
      # By default, the metric is top-1 accuracy. Uncomment the line below fo
      # datasets where metric is mean per-class accuracy.
      # eval_fn=linear_eval.evaluate_per_class
    )
    if "gs://gresearch" in model_dir:
      save_model_dir = model_dir.replace("gs://gresearch/representation-interpretability/breeds", "gs://representation_clustering/previous_models")
    else:
      save_model_dir = model_dir
    gcloud_fs = pyfs.open_fs(save_model_dir)
    with gcloud_fs.open(f'linear_transfer_{layer_name}.pkl', 'wb') as f:
      pickle.dump(out, f)
