**Note:** The folders `.testing`, `.results`, `models`, and `.bigdata` contain experimental files for extensions of this work. Please ignore.

# Description

Tests out federated learning with personalization layers for load forecasting. Uses NREL ComStock dataset to extract load profiles and other auxiliary data. Working on documentation; currently I'm running the code as follows.

```
mpiexec -np 13 --map-by core:PE=1 python train.py --state CA --choice_local 0 # run Prox on client and all server algos
mpiexec -np 13 --map-by core:PE=1 python train.py --state CA --choice_local 1 # run ProxAdam on client and all server algos
mpiexec -np 13 --map-by core:PE=1 python train.py --state CA --choice_local 2 # run Adam on client and all server algos
mpiexec -np 13 --map-by core:PE=1 python train.py --state CA --choice_local 3 # run AdamAMS on client and all server algos
```
Replace `CA` with `IL` or `NY` to test Illinois or New York instead of California. After you run the test (say for CA), then there will appear a folder called `experimentsCA` with plots and a text file `results.txt` which contains final testing MASE errors. Obviously, you will need `MPI` installed on your system. Plus your local python installation and/or env should have `torch` (with cuda) and `mpi4py` installed.

The original source of ComStock data is https://data.openei.org/submissions/4520.

# Client Algorithms
Adam, AdamAMS, ProxAdam, Prox

# Server Algorithms
FedAvg, FedAvgAdaptive, FedAdagradm, FedYogi, FedAdam

