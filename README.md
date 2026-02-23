# TrustFED

This repository provides the official PyTorch implementation of **TrustFED**, a robust federated learning framework designed to detect and defend against label-flipping attacks in both IID and non-IID environments.

## Paper

[Label-Flip Attack Detection via Trust-Weighted Aggregation in Federated Learning for Underground Mine Security](#)  
(*Link will be updated upon publication*)

## Overview

TrustFED introduces a novel trust-weighted aggregation mechanism that dynamically estimates the reliability of client updates based on the historical behavior of their gradient statistics and cluster-based proximity. It is capable of detecting malicious clients launching **label-flipping attacks** and ensures accurate model convergence even under severe data heterogeneity and adversarial presence.

## Datasets

The following datasets are supported and used in the experiments:

- [MNIST](http://yann.lecun.com/exdb/mnist/) — Auto-downloads
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) — Auto-downloads
- [MineSigns]- Newly created dataset from an experimental Mine. The dataset will be made public after publication.  
  Please rename it to `imdb.csv` and place it inside the `data/IMDB/` directory.

## Running the Code

Each dataset has a corresponding Jupyter notebook or Python script that reproduces the results reported in our experiments. Detailed instructions are provided inside each script to configure:

- Data distribution (IID or non-IID via Dirichlet)
- Attack ratio and behavior (label flip attacker configuration)
- Aggregation method selection (FedAvg, Median, Tmean, TrustFED-LFD)

## Dependencies

- Python 3.6+
- PyTorch 1.6+
- TensorFlow 2.x
- scikit-learn
- NumPy, pandas, matplotlib

Install via pip:
```bash
pip install -r requirements.txt
