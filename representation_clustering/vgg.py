from jax import random
import jax.numpy as jnp
import flax.linen as nn
import functools
from typing import Any
import warnings
import functools

class VGG(nn.Module):
    """
    VGG.

    Attributes:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the VGG activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        architecture (str):
            Architecture type:
                - 'vgg16'
                - 'vgg19'
        include_head (bool):
            If True, include the three fully-connected layers at the top of the network.
            This option is useful when you want to obtain activations for images whose
            size is different than 224x224.
        num_classes (int):
            Number of classes. Only relevant if 'include_head' is True.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.
    """
    output: str='softmax'
    pretrained: str='imagenet'
    normalize: bool=True
    architecture: str='vgg16'
    include_head: bool=True
    num_classes: int=1000
    kernel_init: functools.partial=nn.initializers.lecun_normal()
    bias_init: functools.partial=nn.initializers.zeros
    ckpt_dir: str=None
    dtype: str='float32'
    include_bn: bool=False
    include_ln: bool=False
    batch_norm_decay: float = 0.9

    def setup(self):
        print("INCLUDE BN:", self.include_bn)
        self.param_dict = None

    @nn.compact
    def __call__(self, x, train=True):
        """
        Args:
            x (tensor of shape [N, H, W, 3]):
                Batch of input images (RGB format). Images must be in range [0, 1].
                If 'include_head' is True, the images must be 224x224.
            train (bool): Training mode.

        Returns:
            If output == 'logits' or output == 'softmax':
                (tensor): Output tensor of shape [N, num_classes].
            If output == 'activations':
                (dict): Dictionary of activations.
        """
        if self.output not in ['softmax', 'log_softmax', 'logits', 'activations']:
            raise ValueError('Wrong argument. Possible choices for output are "softmax", "logits", and "activations".')

        if self.pretrained is not None and self.pretrained != 'imagenet':
            raise ValueError('Wrong argument. Possible choices for pretrained are "imagenet" and None.')

        if self.include_head and (x.shape[1] != 224 or x.shape[2] != 224):
            raise ValueError('Wrong argument. If include_head is True, then input image must be of size 224x224.')

        if self.normalize:
            mean = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, -1).astype(x.dtype)
            std = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, -1).astype(x.dtype)
            x = (x - mean) / std

        if self.pretrained == 'imagenet':
            if self.num_classes != 1000:
                warnings.warn(f'The user specified parameter \'num_classes\' was set to {self.num_classes} '
                                'but will be overwritten with 1000 to match the specified pretrained checkpoint \'imagenet\', if ', UserWarning)

            num_classes = 1000
        else:
            num_classes = self.num_classes

        act = {}
        if self.include_bn:
            norm = functools.partial(nn.BatchNorm, use_running_average=not train, momentum=self.batch_norm_decay)
        elif self.include_ln:
            norm = nn.LayerNorm
        else:
            norm = None

        x = self._conv_block(x, features=64, num_layers=2, block_num=1, act=act, dtype=self.dtype, bn=norm)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self._conv_block(x, features=128, num_layers=2, block_num=2, act=act, dtype=self.dtype, bn=norm)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self._conv_block(x, features=256, num_layers=3 if self.architecture == 'vgg16' else 4, block_num=3, act=act, dtype=self.dtype, bn=norm)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self._conv_block(x, features=512, num_layers=3 if self.architecture == 'vgg16' else 4, block_num=4, act=act, dtype=self.dtype, bn=norm)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self._conv_block(x, features=512, num_layers=3 if self.architecture == 'vgg16' else 4, block_num=5, act=act, dtype=self.dtype, bn=norm)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        if self.include_head:
            # NCHW format because weights are from pytorch
            x = jnp.transpose(x, axes=(0, 3, 1, 2))
            x = jnp.reshape(x, (-1, x.shape[1] * x.shape[2] * x.shape[3]))
            x = self._fc_block(x, features=4096, block_num=6, relu=True, dropout=True, act=act, train=train, dtype=self.dtype, bn=norm)
            x = self._fc_block(x, features=4096, block_num=7, relu=True, dropout=True, act=act, train=train, dtype=self.dtype, bn=norm)
            x = self._fc_block(x, features=num_classes, block_num=8, relu=False, dropout=False, act=act, train=train, dtype=self.dtype, bn=norm)

        if self.output == 'activations':
            return act 

        if self.output == 'softmax' and self.include_head:
            x = nn.softmax(x)
        if self.output == 'log_softmax' and self.include_head:
            x = nn.log_softmax(x)
        return x

    def _conv_block(self, x, features, num_layers, block_num, act, dtype='float32', bn=None):
        for l in range(num_layers):
            layer_name = f'conv{block_num}_{l + 1}'
            w = self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict[layer_name]['weight']) 
            b = self.bias_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict[layer_name]['bias']) 
            x = nn.Conv(features=features, kernel_size=(3, 3), kernel_init=w, bias_init=b, padding='same', name=layer_name, dtype=dtype)(x)
            act[layer_name] = x
            if bn is not None:
                x = bn(name=f"bn{block_num}_{l+1}")(x)
            x = nn.relu(x)
            act[f'relu{block_num}_{l + 1}'] = x
        return x

    def _fc_block(self, x, features, block_num, act, relu=False, dropout=False, train=True, dtype='float32', bn=None):
        layer_name = f'fc{block_num}'
        w = self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict[layer_name]['weight']) 
        b = self.bias_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict[layer_name]['bias']) 
        x = nn.Dense(features=features, kernel_init=w, bias_init=b, name=layer_name, dtype=dtype)(x)
        act[layer_name] = x
        if bn is not None:
            x = bn(name=f"bn{block_num}")(x)
        if relu:
            x = nn.relu(x)
            act[f'relu{block_num}'] = x
        if dropout: x = nn.Dropout(rate=0.5)(x, deterministic=not train)
        return x  


VGG16 = functools.partial(VGG, output='softmax', pretrained=None, normalize=False, architecture='vgg16',
          include_head=True, kernel_init=nn.initializers.lecun_normal(), bias_init=nn.initializers.zeros,
          ckpt_dir=None, dtype='float32')
VGG19 = functools.partial(VGG, output='softmax', pretrained=None, normalize=False, architecture='vgg19',
          include_head=True, kernel_init=nn.initializers.lecun_normal(), bias_init=nn.initializers.zeros, 
          ckpt_dir=None, dtype='float32')
