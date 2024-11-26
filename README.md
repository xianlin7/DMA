# DMA
This repo is the official implementation for:\
[MICCAI2024] [Revisiting Self-attention in Medical Transformers via Dependency Sparsification](https://link.springer.com/chapter/10.1007/978-3-031-72120-5_52).\
(The details of our DMA can be found at the models directory in this repo or in the paper. We take SETR for example.)

## Requirements
* python 3.6
* pytorch 1.8.0
* torchvision 0.9.0
* more details please see the requirements.txt

## Datasets
* The BTCV (Synapse) dataset could be acquired from [here](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789). 
* The INSTANCE dataset could be acquired from [here](https://instance.grand-challenge.org/).
* The ACDC dataset could be acquired from [here](https://www.creatis.insa-lyon.fr/Challenge/acdc/). 

## Training
Commands for training
```
python train.py
```
## Testing
Commands for testing
``` 
python test.py
```
