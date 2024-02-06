>pyOut.log
mpiexec --use-hwthread-cpus -np 13 --map-by core:PE=1 --report-bindings python train.py --state CA | tee pyOut.log
cp train.py experimentsCA/train.py
cp train.sh experimentsCA/train.sh