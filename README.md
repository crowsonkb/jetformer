# jetformer

Work in progress implementation of [Jetformer](https://arxiv.org/abs/2411.19722).

Differences from the paper:

- Instead of training on 16x16 RGB patches (dimensionality 768), it trains on 8x8 chroma subsampled patches (dimensionality 96). The transformer input/output dimensionality is 24 instead of 64.

- Instead of annealing the noise augmentation level as training progresses, it trains using a constant noise augmentation level of sigma=0.05 (for images in the range -1 to 1) and denoises in inference using Tweedie's formula (see [TarFlow](https://arxiv.org/abs/2412.06329)). (TODO: is this as good as annealing the noise level?)

- The normalizing flow uses 2D sliding window self-attention.
