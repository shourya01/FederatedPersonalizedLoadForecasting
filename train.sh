>pyOut.log
mpiexec --use-hwthread-cpus -np 13 --map-by core:PE=1 --report-bindings python train.py > pyOut.log
cp train.py experiments/train.py