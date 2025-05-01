
![NovaNox](https://github.com/user-attachments/assets/f4b26c51-1bdc-4d00-81a9-5db299563b50)

# Nova-Nox-Neural-Network

## About

N4: Nova-Nox-Neural-Network is a mechanism designed to enhance accuracy by integrating the self-referential capabilities of the Attention mechanism with a simplified version of the selective copying architecture inspired by S6, thereby enabling the acquisition of a more expressive QK matrix. 

The architecture employs ASGG: Adaptive Swish-GELU Gating as the activation function within its MLP components, contributing to richer representational capacity. 

Furthermore, it utilizes DyT for normalization, which improves computational efficiency. 

This repository provides a simplified implementation of N4, and it can be readily integrated with recent Attention-based architectures such as SWA, GQA, and MLA.

***

### Key Features

1. A simplified Selective copying mechanism

2. ASGG: Adaptive Swish-GELU Gating + MLP

3. DyT: Dynamic Tanh Normalization

***

## Implemenation and License

This repository is official pure pytorch implementation.

Licensed under ["MIT License"](https://mit-license.org/).

Commercial use permitted

## How to use

- Clone the repository

```bash
git clone https://github.com/Rikka-Botan/Nova-Nox-Neural-Network.git
```


- Import necessary libraries

```python
import torch
from torch import nn
import torch.nn.functional as F
from model.N4_modeling import N4C
```


- Model create

```python
"""
Args:
hidden_size: int - model hidden size,
inter_size: int - model mlp intermediate size,
vocab_size : int - tokenizer vocab num,
heads: int - heads num,
layers: int - N4D(Decoder) layers num
"""

hidden_size = 768
intermediate_size = 3072
vocab_size = 32064
heads = 6
layers = 6

model = N4C(
  hidden_size,
  intermediate_size,
  vocab_size,
  heads,
  layers
)
output = model(tokenized_text)
```


## How to Train

- training code

```python
from torch.optim import AdamW

optimizer = AdamW(
  model.parameters(),
  lr=6.0e-4,
  betas=(0.9, 0.95),
  eps=1e-8,
  weight_decay=1e-1
)

for batch in dataloader:
  optimizer.zero_grad()
  batch = batch.to(device)
  loss = model.to(device)(input=batch, labels=batch)[1]
  loss.backward()
  optimizer.step()
```



## Acknowledgements

I thank the developers of python and pytorch.

I thank all the researchers for their efforts to date.

I thank Japan's high standard of education.

And most of all, thank you for your interest in this repository.

## Citations

I would be happy to include a citation at the end, but it is not required.

Feel free to use this model.


## Contact Us

[My X account](https://x.com/peony__snow)


## About Author

### Rikka Botan

Japanese independent researcher having shy and pampered personality >_<

Twin-tail hair is a charm point :)

Interested in natural language processings. 

Usually using python and C.

![RikkaBotan_Logo](https://github.com/user-attachments/assets/92913f91-9136-4d44-8b4d-8a2120118a05)
