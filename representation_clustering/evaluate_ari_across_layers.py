import fs as pyfs
import pickle
from sklearn.metrics import adjusted_rand_score
import numpy as np
from itertools import combinations

overclustering_factors = [1,2,3,4,5]

if __name__ == "__main__":
  model_dir = 'gs://representation_clustering/entity13_4_subclasses_vgg16'
  n_class = 13
  assert(str(n_class) in model_dir)

  if 'vgg' in model_dir:
    layer_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 
                        'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7']
  else:
    layer_names = ['stage1_block1', 'stage1_block2', 'stage1_block3', 'stage2_block1', 'stage2_block2', \
                   'stage2_block3', 'stage2_block4', 'stage3_block1', 'stage3_block2', 'stage3_block3', \
                   'stage3_block4', 'stage3_block5', 'stage3_block6', 'stage4_block1', 'stage4_block2', 'stage4_block3']
  n_layer = len(layer_names)
  result_dict = {ocf: np.ones((n_layer, n_layer)) for ocf in overclustering_factors}
  all_layer_idx_pairs = list(combinations(list(range(n_layer)), 2))
  gcloud_fs = pyfs.open_fs(model_dir)

  for idx1, idx2 in all_layer_idx_pairs:
    layer1 = layer_names[idx1]
    layer2 = layer_names[idx2]
    print(layer1, layer2)
    with gcloud_fs.open(f'clf_labels_ckpt_{layer1}_normalized.pkl', 'rb') as f:
      data1 = pickle.load(f)
    with gcloud_fs.open(f'clf_labels_ckpt_{layer2}_normalized.pkl', 'rb') as f:
      data2 = pickle.load(f)
    for ocf in overclustering_factors:
      all_class_aris = []
      clf_labels1 = data1[ocf]
      clf_labels2 = data2[ocf]
      for class_idx in range(n_class):
        all_class_aris.append(adjusted_rand_score(clf_labels1[class_idx], clf_labels2[class_idx]))
      result_dict[ocf][idx1][idx2] = np.mean(all_class_aris)
      result_dict[ocf][idx2][idx1] = np.mean(all_class_aris)

  with gcloud_fs.open('aris_across_layers.pkl', 'wb') as f:
    pickle.dump(result_dict, f)
