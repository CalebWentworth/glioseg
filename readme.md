# GlioSeg

A Glioma segmentation model

___

## Requirments

Requirments are found in requirements.txt

Use an virtual env to install the requirments (Conda or mamba works well)
Some requirmentes are only available on the conda forge channel.
___

## Status

- Base model definition is located in /src/network.py
- SSNunet.ipynb is designed for use with the UCSF-PDGM dataset
- BraTS-PED dataset selected as new dataset as the challenge requirements are very specific that models may ONLY be trained on the PEDs dataset, thus a network to network comparison is guarenteed to be fair.
- Spiking_unet_best.pt is trained on the UCSF-PDGM dataset

### Current tasks

- Refactor to remove hardcoded dependency for patch building and data set.

- Refactor testing and debugging tools.

- Ensure compatibility with [BraTS pediatric glioma dataset](https://www.cancerimagingarchive.net/collection/brats-peds/) and [challenge requirements](https://www.synapse.org/Synapse:syn64153130/wiki/631455) e.g. multi label segmentation for tumor regions instead of reduction to a binary classification task
