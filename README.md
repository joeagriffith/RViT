A simple augmentation of PyTorch's VisionTransform class from torchvision.models to include registers, as per: https://arxiv.org/abs/2309.16588

Introduces registers to the encoder that are appended as tokens to the 'patchified' sequence, and excluded in the output.
The tokens are learnable parameters and do not receive positional embeddings.

The API of the class is identical with VisionTransformer, except the additional init argument for 'num_registers', which specifies the number of register tokens.

## Installation
```
pip install rvit
```

## Usage
```
from rvit import RegisteredVisionTransformer
```
