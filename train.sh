>pyOut.log
# mpiexec --use-hwthread-cpus -np 13 --map-by core:PE=1 --report-bindings python train.py --state CA --choice_local 0 | tee pyOut.log
# mpiexec --use-hwthread-cpus -np 13 --map-by core:PE=1 --report-bindings python train.py --state CA --choice_local 1 | tee pyOut.log
# mpiexec --use-hwthread-cpus -np 13 --map-by core:PE=1 --report-bindings python train.py --state CA --choice_local 2 | tee pyOut.log
# mpiexec --use-hwthread-cpus -np 13 --map-by core:PE=1 --report-bindings python train.py --state CA --choice_local 3 | tee pyOut.log
STATE="CA"

mpiexec --use-hwthread-cpus -np 13 --map-by core:PE=1 --report-bindings python train.py --state $STATE --choice_local 0 | tee pyOut.log
wait
mpiexec --use-hwthread-cpus -np 13 --map-by core:PE=1 --report-bindings python train.py --state $STATE --choice_local 1 | tee pyOut.log
wait
mpiexec --use-hwthread-cpus -np 13 --map-by core:PE=1 --report-bindings python train.py --state $STATE --choice_local 2 | tee pyOut.log
wait
mpiexec --use-hwthread-cpus -np 13 --map-by core:PE=1 --report-bindings python train.py --state $STATE --choice_local 3 | tee pyOut.log
wait
cp train.py experimentsCA/train.py
cp train.sh experimentsCA/train.sh