# Description

Tests out federated learning with personalization layers for load forecasting. Uses NREL ComStock dataset to extract load profiles and other auxiliary data. Working on documentation; currently you can run by having PyTorch, mpi4py and NumpPy installed locally or in a virtual environment and running `train.py`.

An alternative approach is using 'prox'-style algorithms to do local updates. The local algorithms are Adam, AdamAMS (regular) and Prox, ProxAdam (prox-style).

# Client Algorithms
Adam, AdamAMS, ProxAdam, Prox

# Server Algorithms
FedAvg, FedAvgAdaptive, FedAdagradm, FedYogi, FedAdam

