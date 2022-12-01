from typing import Any, Sequence
import ml_collections
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import functools
import itertools
from scipy.stats import mode
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import time
import pickle
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

from clu import preprocess_spec
from scipy.special import comb

from absl import logging
from train import create_train_state, cosine_decay
from configs.default_breeds import get_config
from input_pipeline_breeds import RescaleValues, ResizeSmall, CentralCrop, GeneralPreprocessOp
import os
import sys
import fs as pyfs
from flax.training import checkpoints


def get_learning_rate(step: int,
                      *,
                      base_learning_rate: float,
                      steps_per_epoch: int,
                      num_epochs: int,
                      warmup_epochs: int = 5):
  """Cosine learning rate schedule."""
  logging.info(
      "get_learning_rate(step=%s, base_learning_rate=%s, steps_per_epoch=%s, num_epochs=%s",
      step, base_learning_rate, steps_per_epoch, num_epochs)
  if steps_per_epoch <= 0:
    raise ValueError(f"steps_per_epoch should be a positive integer but was "
                     f"{steps_per_epoch}.")
  if warmup_epochs >= num_epochs:
    raise ValueError(f"warmup_epochs should be smaller than num_epochs. "
                     f"Currently warmup_epochs is {warmup_epochs}, "
                     f"and num_epochs is {num_epochs}.")
  epoch = step / steps_per_epoch
  lr = cosine_decay(base_learning_rate, epoch - warmup_epochs,
                    num_epochs - warmup_epochs)
  warmup = jnp.minimum(1., epoch / warmup_epochs)
  return lr * warmup


def predict(model, state, batch):
  """Get intermediate representations from a model."""
  variables = {
      "params": state.ema_params,
      "batch_stats": state.batch_stats
  }
  _, state = model.apply(variables, batch['image'], capture_intermediates=True, mutable=["intermediates"], train=False)
  intermediates = state['intermediates']#['stage4']['__call__'][0]
  return intermediates


def compute_purity(clusters, classes):
  """Compute purity of the cluster."""
  n_cluster_points = 0
  for cluster_idx in set(clusters):
    instance_idx = np.where(clusters == cluster_idx)[0]
    subclass_labels = classes[instance_idx]
    mode_stats = mode(subclass_labels)
    n_cluster_points += mode_stats[1][0]
  purity = n_cluster_points / len(clusters)
  return purity


def evaluate_purity(eval_dataset, model_dir, ckpt_number, n_classes, overcluster_factors, n_subclasses, ood_dataset, normalize_embeddings):
  """Given a model and a dataset, cluster the second-to-last layer representations and compute average purity."""
  config = get_config()
  learning_rate_fn = functools.partial(
      get_learning_rate,
      base_learning_rate=0.1,
      steps_per_epoch=40,
      num_epochs=config.num_epochs,
      warmup_epochs=config.warmup_epochs)
  checkpoint_path = os.path.join(model_dir, f'checkpoints-0/ckpt-{ckpt_number}.flax')
  model, state = create_train_state(config, jax.random.PRNGKey(0), input_shape=(8, 224, 224, 3), 
                                    num_classes=n_classes, learning_rate_fn=learning_rate_fn)
  state = checkpoints.restore_checkpoint(checkpoint_path, state)
  print("Ckpt number", ckpt_number, "Ckpt step:", state.step)

  result_dict = {}
  all_intermediates = []
  all_subclass_labels = []
  all_images = []
  for step, batch in enumerate(eval_ds):
    if step % 50 == 0:
      print(step)
    intermediates = predict(model, state, batch)
    labels = batch['label'].numpy()
    bs = labels.shape[0]
    all_subclass_labels.append(labels)
    all_images.append(batch['image'].numpy())
    all_intermediates.append(np.mean(intermediates['stage4']['__call__'][0], axis=(1,2)).reshape(bs, -1))

  all_intermediates = np.vstack(all_intermediates)
  if normalize_embeddings:
    all_intermediates = normalize(all_intermediates, axis=1, copy=False)
  all_subclass_labels = np.hstack(all_subclass_labels)
  all_images = np.vstack(all_images)

  for overcluster_factor in overcluster_factors:
    clf = AgglomerativeClustering(n_clusters=n_subclasses*overcluster_factor,
                                          linkage='ward').fit(all_intermediates)
    all_clf_labels = clf.labels_
    purity = compute_purity(clf.labels_, all_subclass_labels)
    result_dict[overcluster_factor] = purity

  if "gs://gresearch" in model_dir:
    model_dir = model_dir.replace("gs://gresearch/representation-interpretability/breeds", "gs://representation_clustering/previous_models")
  gcloud_fs = pyfs.open_fs(model_dir)
  out_file = f'{ood_dataset}_purity.pkl'
  if normalize_embeddings:
    out_file = out_file.replace('.pkl', '_normalized.pkl')
  with gcloud_fs.open(out_file, 'wb') as f:
    pickle.dump(result_dict, f)
  return result_dict, all_clf_labels


if __name__ == "__main__":
  DATASET = sys.argv[1]
  n_subclasses = int(sys.argv[2])
  if DATASET == "tf_flowers" or DATASET == "uc_merced":
    print(DATASET, n_subclasses, 'train split')
    SPLIT = tfds.Split.TRAIN
  else:
    print(DATASET, n_subclasses, 'test_split')
    SPLIT = tfds.Split.TEST

  dataset_builder = tfds.builder(DATASET, try_gcs=True)
  dataset_builder.download_and_prepare()
  eval_preprocess = preprocess_spec.PreprocessFn([
    RescaleValues(),
    ResizeSmall(256),
    CentralCrop(224),
    GeneralPreprocessOp(),
  ], only_jax_types=True)
  dataset_options = tf.data.Options()
  dataset_options.experimental_optimization.map_parallelization = True
  dataset_options.experimental_threading.private_threadpool_size = 48
  dataset_options.experimental_threading.max_intra_op_parallelism = 1
  read_config = tfds.ReadConfig(shuffle_seed=None, options=dataset_options)
  eval_ds = dataset_builder.as_dataset(
    split=SPLIT,
    shuffle_files=False,
    read_config=read_config,
    decoders=None)
  batch_size = 8
  eval_ds = eval_ds.cache()
  eval_ds = eval_ds.map(eval_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  eval_ds = eval_ds.batch(batch_size, drop_remainder=False)
  eval_ds = eval_ds.prefetch(tf.data.experimental.AUTOTUNE)

  model_metadata = [("gs://representation_clustering/entity13_fine_grained", 81, 260),
                  ("gs://representation_clustering/living17_fine_grained", 81, 68),
                  ("gs://representation_clustering/nonliving26_fine_grained", 81, 104),
                  ]

  BASE_DIR = "gs://gresearch/representation-interpretability/breeds/"
  #model_metadata = [(os.path.join(BASE_DIR, 'entity13_400_epochs_ema_0.99_bn_0.99/'), 161, 13),
  #                (os.path.join(BASE_DIR, 'living17_400_epochs_ema_0.99_bn_0.99/'), 173, 17),
  #                (os.path.join(BASE_DIR, 'nonliving26_400_epochs_ema_0.99_bn_0.99/'), 257, 26),
  #                (os.path.join(BASE_DIR, 'imagenet_ema_0.99_bn_0.99/'), 8, 1000)
  #                ]

  overcluster_factors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  normalize_embeddings = True

  for model_dir, ckpt_number, N_CLASSES in model_metadata:
    print(model_dir)
    result_dict, _ = evaluate_purity(eval_ds, model_dir, ckpt_number, N_CLASSES, overcluster_factors, n_subclasses, DATASET, normalize_embeddings)
    print(result_dict)
