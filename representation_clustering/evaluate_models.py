import os
import itertools
import functools
import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
from clu import preprocess_spec
from train import create_train_state
from compute_purity_ami import load_eval_ds, get_learning_rate
from flax.training import checkpoints
import breeds_helpers
from configs.default_breeds import get_config
import numpy as np
import pickle
import fs as pyfs


def predict(model, state, batch):
  """Get intermediate representations from a model."""
  variables = {
      "params": state.params if state.ema_params is None else state.ema_params,
      "batch_stats": state.batch_stats
  }
  logits = model.apply(variables, batch["image"], mutable=False, train=False)
  predictions = jnp.argmax(logits, axis=-1)
  acc = np.asarray(predictions == batch["label"]).mean()
  #correct_preds = np.asarray(predictions == batch["label"])
  return acc, np.asarray(predictions)


def eval_model(model_dir, ckpt_number, dataset_type, num_classes, num_subclasses, shuffle_subclasses):
  checkpoint_path = os.path.join(model_dir, f'checkpoints-0/ckpt-{ckpt_number}.flax')
  eval_ds, num_classes, train_subclasses = load_eval_ds(dataset_type, num_classes, num_subclasses, shuffle_subclasses, lookup_labels=True)
  config = get_config()
  learning_rate_fn = functools.partial(
      get_learning_rate,
      base_learning_rate=0.1,
      steps_per_epoch=40,
      num_epochs=config.num_epochs,
      warmup_epochs=config.warmup_epochs)
  model, state = create_train_state(config, jax.random.PRNGKey(0), input_shape=(8, 224, 224, 3), num_classes=num_classes, learning_rate_fn=learning_rate_fn)
  
  state = checkpoints.restore_checkpoint(checkpoint_path, state)
  print("Load model ckpt with step:", state.step)
  all_accs, batch_sizes = [], []
  all_preds = []
  all_labels = []

  for step, batch in enumerate(eval_ds):
    acc, preds = predict(model, state, batch)
    n_examples = batch['label'].numpy().shape[0]
    all_labels.extend(list(batch['label'].numpy()))
    all_accs.append(acc)
    all_preds.extend(list(preds))
    batch_sizes.append(n_examples)
  print(all_accs)
  acc = sum([a*b for a,b in zip(all_accs, batch_sizes)]) / sum(batch_sizes)
  print(f"Accuracy of the model: {acc*100}%")
  return acc, all_preds, all_labels


if __name__ == "__main__":
  evaluate_all_epochs = False
  dataset_types = ['living17', 'nonliving26', 'entity13_4_subclasses', 'entity13_4_subclasses_shuffle'] #entity13
  #ckpt_numbers = [173, 257, 161, 129, 129]
  ckpt_numbers = [81]*5
  for i, dataset_type in enumerate(dataset_types):
    #if 'nonliving' in dataset_type:
    #  continue
    print(f"################### {dataset_type} ###################")
    og_dataset_type = dataset_type
    #model_dir = f"gs://gresearch/representation-interpretability/breeds/{dataset_type}_400_epochs_ema_0.99_bn_0.99/"
    model_dir = f"gs://representation_clustering/{dataset_type}_vgg16"
    num_classes = -1
    num_subclasses = -1
    shuffle_subclasses = False
    if 'shuffle' in dataset_type:
      shuffle_subclasses = True
    if '4_subclasses' in dataset_type:
      dataset_type = dataset_type.split('_')[0]
      num_subclasses = 4
    print(model_dir)
    ckpt_number = ckpt_numbers[i]
    if not evaluate_all_epochs:
      acc, all_preds, all_labels = eval_model(model_dir, ckpt_number, dataset_type, num_classes, num_subclasses, shuffle_subclasses)
      print(all_preds)
      results = {'accuracy': acc, 'preds': all_preds, 'labels': all_labels}
      print("")
    else:
      results = {}
      for ckpt in range(int(ckpt_number/4), ckpt_number):
        print("CKPT:", ckpt)
        acc, _, _ = eval_model(model_dir, ckpt, dataset_type, num_classes, num_subclasses, shuffle_subclasses)
        results[ckpt] = acc
      print("ALL ACCURACIES:", results)
    if "gs://gresearch" in model_dir:
      model_dir = model_dir.replace("gs://gresearch/representation-interpretability/breeds", "gs://representation_clustering/previous_models")
    gcloud_fs = pyfs.open_fs(model_dir)
    with gcloud_fs.open("ckpt_val_acc.pkl", 'wb') as f:
      pickle.dump(results, f)
