A simple augmentation of PyTorch's VisionTransform class from torchvision.models to include registers, as per: https://arxiv.org/abs/2309.16588

Introduces registers to the encoder that are appended as tokens to the 'patchified' sequence, and excluded in the output.
The tokens are learnable parameters and do not receive positional embeddings.

The API of the class is identical with VisionTransformer, except the additional init argument for 'num_registers', which specifies the number of register tokens.

## Installation
```
pip install rvit
```

## Initialisation
```
rvit.RegisteredVisionTransformer(
  image_size: int,
  patch_size: int,
  num_layers: int,
  num_heads: int,
  hidden_dim: int,
  num_registers: int, # Specifies number of register tokens
  mlp_dim: int,
  dropout: float = 0.0,
  attention_dropout: float = 0.0,
  num_classes:int = 1000,
  representation_size: Optional[int] = None,
  norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
  conv_stem_configs: Optional[List[ConvStemConfig]] = None,
)
```
