import os
import itertools
import functools
import tensorflow as tf
import tensorflow_datasets as tfds
import jax
from clu import preprocess_spec
from train import create_train_state
from input_pipeline_breeds import predicate, RescaleValues, ResizeSmall, CentralCrop
from flax.training import checkpoints
import breeds_helpers
from configs.default_breeds import get_config
import numpy as np
from scipy.stats import mode
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score
import pickle
import fs as pyfs


BREEDS_INFO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "breeds")


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


@functools.partial(jax.jit, static_argnums=(0,))
def predict(model, state, batch):
  """Get intermediate representations from a model."""
  #print(state.params)
  #print('---------')
  #print(state.batch_stats)
  variables = {
      "params": state.params if state.ema_params is None else state.ema_params,
      "batch_stats": state.batch_stats
  }
  _, state = model.apply(variables, batch['image'], capture_intermediates=True, mutable=["intermediates"], train=False)
  intermediates = state['intermediates']
  return intermediates #maybe return only the relevant intermediates


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


def load_eval_ds(dataset_type, num_classes, num_subclasses, shuffle_subclasses):
  ret = breeds_helpers.make_breeds_dataset(
      dataset_type,
      BREEDS_INFO_DIR,
      BREEDS_INFO_DIR,
      num_classes=num_classes,
      num_subclasses=num_subclasses,
      shuffle_subclasses=shuffle_subclasses,
      split=None)
  superclasses, subclass_split, label_map = ret
  train_subclasses = subclass_split[0]
  num_classes = len(train_subclasses)
  all_subclasses = list(itertools.chain(*train_subclasses))
  new_label_map = {}
  for subclass_idx, sub in enumerate(all_subclasses):
    new_label_map.update({sub: subclass_idx})
  print(new_label_map)
  lookup_table = tf.lookup.StaticHashTable(
      initializer=tf.lookup.KeyValueTensorInitializer(
          keys=tf.constant(list(new_label_map.keys()), dtype=tf.int64),
          values=tf.constant(list(new_label_map.values()), dtype=tf.int64),
      ),
      default_value=tf.constant(-1, dtype=tf.int64))

  dataset_builder = tfds.builder("imagenet2012:5.0.0", try_gcs=True)
  eval_preprocess = preprocess_spec.PreprocessFn([
      RescaleValues(),
      ResizeSmall(256),
      CentralCrop(224),
      #LabelMappingOp(lookup_table=lookup_table)
      ], only_jax_types=True)


  dataset_options = tf.data.Options()
  dataset_options.experimental_optimization.map_parallelization = True
  dataset_options.experimental_threading.private_threadpool_size = 48
  dataset_options.experimental_threading.max_intra_op_parallelism = 1

  read_config = tfds.ReadConfig(shuffle_seed=None, options=dataset_options)
  eval_ds = dataset_builder.as_dataset(
      split=tfds.Split.VALIDATION,
      shuffle_files=False,
      read_config=read_config,
      decoders=None)

  batch_size = 32
  eval_ds = eval_ds.filter(functools.partial(predicate, all_subclasses=all_subclasses))
  eval_ds = eval_ds.cache()
  eval_ds = eval_ds.map(eval_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  eval_ds = eval_ds.batch(batch_size, drop_remainder=False)
  eval_ds = eval_ds.prefetch(tf.data.experimental.AUTOTUNE)
  return eval_ds, num_classes, train_subclasses


def evaluate_purity_across_layers():
  dataset_type = "entity13"
  eval_ds, num_classes, train_subclasses = load_eval_ds(dataset_type, -1, -1, False)
  config = get_config()
  learning_rate_fn = functools.partial(
      get_learning_rate,
      base_learning_rate=0.1,
      steps_per_epoch=40,
      num_epochs=config.num_epochs,
      warmup_epochs=config.warmup_epochs)
  model_dir = f"gs://representation_clustering/{dataset_type}_vgg16/"
  checkpoint_path = os.path.join(model_dir, 'checkpoints-0/ckpt-81.flax')
  model, state = create_train_state(config, jax.random.PRNGKey(0), input_shape=(8, 224, 224, 3), num_classes=num_classes, learning_rate_fn=learning_rate_fn)
  state = checkpoints.restore_checkpoint(checkpoint_path, state)
  print("step:", state.step)

  metric_name = "ami"
  for stage_prefix in ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3', 'conv4', 'conv5', 'fc']:
    all_layer_intermediates = {}
    all_subclass_labels = []
    for step, batch in enumerate(eval_ds):
      if step % 50 == 0:
        print("Step:", step)
      intermediates = predict(model, state, {k: jax.numpy.asarray(v) for k, v in batch.items()})
      labels = batch['label'].numpy()
      bs = labels.shape[0]
      all_subclass_labels.append(labels)
    
      count = 0
      for layer in sorted(intermediates.keys()):
        if not layer.startswith(stage_prefix):
          continue
        key = layer
        if key not in all_layer_intermediates:
          all_layer_intermediates[key] = []
        all_layer_intermediates[key].append(np.array(intermediates[key]['__call__'][0]).reshape(bs, -1))
    for k, v in all_layer_intermediates.items():
      print(k, v[0].shape)

    all_subclass_labels = np.hstack(all_subclass_labels)
    print(all_subclass_labels.shape)

    for key, all_intermediates in all_layer_intermediates.items():
      n_subclasses = len(train_subclasses[0])
      all_intermediates = np.vstack(all_intermediates)
      print(key, all_intermediates.shape)
      result_dict = {}

      for overcluster_factor in [1, 2, 3, 4, 5]:
        all_clfs = []

        for subclasses in train_subclasses:
          subclass_idx = np.array([i for i in range(len(all_subclass_labels)) if all_subclass_labels[i] in subclasses])
          hier_clustering = AgglomerativeClustering(n_clusters=len(subclasses)*overcluster_factor,
                                                linkage='ward').fit(all_intermediates[subclass_idx])
          all_clfs.append(hier_clustering)


        metric_list = []
        all_clf_labels = []
        for i, clf in enumerate(all_clfs):
          all_clf_labels.append(clf.labels_)
          subclasses = train_subclasses[i]
          subclass_idx = np.array([
              i for i in range(len(all_subclass_labels))
              if all_subclass_labels[i] in subclasses
          ])
          subclass_labels = all_subclass_labels[subclass_idx]
          if metric_name == 'purity':
            metric = compute_purity(clf.labels_, subclass_labels)
          elif metric_name == 'ami':
            metric = adjusted_mutual_info_score(subclass_labels, clf.labels_)
          metric_list.append(metric)

        result_dict[overcluster_factor] = metric_list

      print(result_dict)
      gcloud_fs = pyfs.open_fs(model_dir)
      if metric_name == "purity":
        metric_outfile = f'class_purity_ckpt_{key}.pkl'
        clf_label_outfile = f'clf_labels_ckpt_{key}.pkl'
        with gcloud_fs.open(metric_outfile, 'wb') as f:
          pickle.dump(metric_list, f)
        with gcloud_fs.open(clf_label_outfile, 'wb') as f:
          pickle.dump(all_clf_labels, f)
      else:
        metric_outfile = f'adjusted_mutual_info_ckpt_{key}.pkl'
        with gcloud_fs.open(metric_outfile, 'wb') as f:
          pickle.dump(result_dict, f)

if __name__ == "__main__":
  evaluate_purity_across_layers()
