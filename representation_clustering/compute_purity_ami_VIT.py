import os
import itertools
import functools
import tensorflow as tf
import tensorflow_datasets as tfds
import jax
from clu import preprocess_spec
from train import create_train_state
from input_pipeline_breeds import predicate, RescaleValues, ResizeSmall, CentralCrop, LabelMappingOp
from flax.training import checkpoints
import breeds_helpers
from configs.default_breeds import get_config
import numpy as np
from scipy.stats import mode
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import normalize
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
  _, intermediates = model.apply(variables, batch['image'], mutable=False, train=False) #capture_intermediates=True, mutable=["intermediates"], train=False)
  #intermediates = state['intermediates']
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


def load_eval_ds(dataset_type, num_classes, num_subclasses, shuffle_subclasses, split=tfds.Split.VALIDATION, lookup_labels=False, use_fine_grained_labels=False):
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
  for super_idx, sub in enumerate(train_subclasses):
    new_label_map.update({s: super_idx for s in sub})

  if use_fine_grained_labels:
    num_classes = len(all_subclasses)
    for super_idx, sub in enumerate(all_subclasses):
      new_label_map.update({sub : super_idx})

  print(new_label_map)
  lookup_table = tf.lookup.StaticHashTable(
      initializer=tf.lookup.KeyValueTensorInitializer(
          keys=tf.constant(list(new_label_map.keys()), dtype=tf.int64),
          values=tf.constant(list(new_label_map.values()), dtype=tf.int64),
      ),
      default_value=tf.constant(-1, dtype=tf.int64))

  dataset_builder = tfds.builder("imagenet2012:5.0.0", try_gcs=True)
  if lookup_labels:
    eval_preprocess = preprocess_spec.PreprocessFn([
          RescaleValues(),
          ResizeSmall(256),
          CentralCrop(224),
          LabelMappingOp(lookup_table=lookup_table)
          ], only_jax_types=True)
  else:
    eval_preprocess = preprocess_spec.PreprocessFn([
      RescaleValues(),
      ResizeSmall(256),
      CentralCrop(224),
      ], only_jax_types=True)

  dataset_options = tf.data.Options()
  dataset_options.experimental_optimization.map_parallelization = True
  dataset_options.experimental_threading.private_threadpool_size = 48
  dataset_options.experimental_threading.max_intra_op_parallelism = 1

  read_config = tfds.ReadConfig(shuffle_seed=None, options=dataset_options)
  eval_ds = dataset_builder.as_dataset(
      split=split,
      shuffle_files=False,
      read_config=read_config,
      decoders=None)

  batch_size = 16
  eval_ds = eval_ds.filter(functools.partial(predicate, all_subclasses=all_subclasses))
  eval_ds = eval_ds.cache()
  eval_ds = eval_ds.map(eval_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  eval_ds = eval_ds.batch(batch_size, drop_remainder=False)
  eval_ds = eval_ds.prefetch(tf.data.experimental.AUTOTUNE)
  return eval_ds, num_classes, train_subclasses


def evaluate_purity_across_layers(ckpt_number, seed=1):
  dataset_type = "entity13_4_subclasses_shuffle"
  model_dir = f"gs://representation_clustering/{dataset_type}_vit_tok/"
  print(f"################## EVALUATING ckpt_number={ckpt_number}, seed={seed} #################")

  if '4_subclasses' in model_dir:
    num_subclasses = 4
  else:
    num_subclasses = -1
  if "shuffle" in model_dir:
    shuffle_subclasses = True
  else:
    shuffle_subclasses = False 
  if 'fine_grained' in model_dir:
    use_fine_grained_labels = True
  else:
    use_fine_grained_labels = False

  dataset_type = dataset_type.split('_')[0] 
  eval_ds, num_classes, train_subclasses = load_eval_ds(dataset_type, -1, num_subclasses, shuffle_subclasses, use_fine_grained_labels=use_fine_grained_labels)
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

  normalize_embeddings = True
  print(f"-------------------------- EVALUATING, normalize_embeddings={normalize_embeddings}")
  if 'vgg' in model_dir:
    layer_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 
                        'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
  if 'vit' in model_dir:
    layer_names = ['block00_+mlp', ]
  else:
    layer_names = ['stage1_block1', 'stage1_block2', 'stage1_block3', 'stage2_block1', 'stage2_block2', \
                   'stage2_block3', 'stage2_block4', 'stage3_block1', 'stage3_block2', 'stage3_block3', \
                   'stage3_block4', 'stage3_block5', 'stage3_block6', 'stage4_block1', 'stage4_block2', 'stage4_block3']

  layer_names = []
  for block_no in range(12):
    for suffix in ['+mlp', '+sa']:
      layer_names.append(f'block{block_no:02d}_{suffix}')

  for layer in layer_names:
    print('########################', layer)
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
      key = layer
      if 'vgg' in model_dir or layer == 'head':
        if key not in all_layer_intermediates:
          all_layer_intermediates[key] = []
        all_layer_intermediates[key].append(np.array(intermediates[key]['__call__'][0]).reshape(bs, -1))
      else:
        block, suffix = layer.split('_')
        if key not in all_layer_intermediates:
          all_layer_intermediates[key] = []
        all_layer_intermediates[key].append(np.array(intermediates['encoder'][block][suffix]).reshape(bs, -1)) 
    for k, v in all_layer_intermediates.items():
      print(k, v[0].shape)

    all_subclass_labels = np.hstack(all_subclass_labels)
    print(all_subclass_labels.shape)
    

    for key, all_intermediates in all_layer_intermediates.items():
      n_subclasses = len(train_subclasses[0])
      all_intermediates = np.vstack(all_intermediates)
      if normalize_embeddings:
        all_intermediates = normalize(all_intermediates, axis=1, copy=False)
      print(key, all_intermediates.shape)
      purity_result_dict, ami_result_dict = {}, {}
      clf_labels_dict = {}

      for overcluster_factor in [1, 2, 3, 4, 5]:
        all_clfs = []

        for subclasses in train_subclasses:
          subclass_idx = np.array([i for i in range(len(all_subclass_labels)) if all_subclass_labels[i] in subclasses])
          hier_clustering = AgglomerativeClustering(n_clusters=len(subclasses)*overcluster_factor,
                                                linkage='ward').fit(all_intermediates[subclass_idx])
          all_clfs.append(hier_clustering)


        purity_metric_list, ami_metric_list, clf_labels_list = [], [], []
        all_clf_labels = []
        for i, clf in enumerate(all_clfs):
          all_clf_labels.append(clf.labels_)
          subclasses = train_subclasses[i]
          subclass_idx = np.array([
              i for i in range(len(all_subclass_labels))
              if all_subclass_labels[i] in subclasses
          ])
          subclass_labels = all_subclass_labels[subclass_idx]
          purity_metric = compute_purity(clf.labels_, subclass_labels)
          ami_metric = adjusted_mutual_info_score(subclass_labels, clf.labels_)
          purity_metric_list.append(purity_metric)
          ami_metric_list.append(ami_metric)
          clf_labels_list.append(clf.labels_)

        purity_result_dict[overcluster_factor] = purity_metric_list
        ami_result_dict[overcluster_factor] = ami_metric_list
        clf_labels_dict[overcluster_factor] = clf_labels_list

      print(purity_result_dict)
      print(ami_result_dict)
      if "gs://gresearch" in model_dir:
        save_model_dir = model_dir.replace("gs://gresearch/representation-interpretability/breeds", "gs://representation_clustering/previous_models")
      else:
        save_model_dir = model_dir
      gcloud_fs = pyfs.open_fs(save_model_dir)
      
      class_purity_file = f'class_purity_ckpt_{key}.pkl'
      clf_labels_file = f'clf_labels_ckpt_{key}.pkl'
      ami_file = f'adjusted_mutual_info_ckpt_{key}.pkl'
      if normalize_embeddings:
        class_purity_file = class_purity_file.replace('.pkl', '_normalized.pkl')
        clf_labels_file = clf_labels_file.replace('.pkl', '_normalized.pkl')
        ami_file = ami_file.replace('.pkl', '_normalized.pkl')
      if ckpt_number != 81:
        class_purity_file = class_purity_file.replace('.pkl', f'_ckpt_{ckpt_number}.pkl')
        clf_labels_file = clf_labels_file.replace('.pkl', f'_ckpt_{ckpt_number}.pkl')
        ami_file = ami_file.replace('.pkl', f'_ckpt_{ckpt_number}.pkl')
      print(class_purity_file, clf_labels_file, ami_file)
      with gcloud_fs.open(class_purity_file, 'wb') as f:
        pickle.dump(purity_result_dict, f)
      with gcloud_fs.open(clf_labels_file, 'wb') as f:
        pickle.dump(clf_labels_dict, f)  
      with gcloud_fs.open(ami_file, 'wb') as f:
        pickle.dump(ami_result_dict, f)


if __name__ == "__main__":
  for ckpt_number in [81]:#[1,21,41,61]:
    for seed in [0]:
      evaluate_purity_across_layers(ckpt_number, seed=seed)
