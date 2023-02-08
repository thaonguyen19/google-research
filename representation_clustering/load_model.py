from configs.default_breeds import get_config
from train import create_train_state
import jax
import functools
import os
import ml_collections
from flax.training import checkpoints

if __name__ == "__main__":
  model_dir = 'gs://representation_clustering/entity13_4_subclasses_vgg16_with_bn/'
  checkpoint_path = os.path.join(model_dir, 'checkpoints-0/ckpt-81.flax')

  rng = jax.random.PRNGKey(42)
  rng, model_rng = jax.random.split(rng)

  config = get_config()
  learning_rate_fn = lambda x: 0.1

  model, state = create_train_state(config, model_rng, input_shape=(8, 224, 224, 3), num_classes=13, learning_rate_fn=learning_rate_fn)
  state = checkpoints.restore_checkpoint(checkpoint_path, state)
  print("step:", state.step)
  print(state)
  print(state.batch_stats)
