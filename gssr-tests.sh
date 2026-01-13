#!/bin/bash

set -x
#set -e

./gssr-record -h
./gssr-record --help
./gssr-record --version
./gssr-record --help -o test-report-00

./gssr-analyze.py -i DOES_NOT_EXIST
./gssr-analyze.py -i test-report-01
stat report.pdf

./gssr-record -o -- sleep
./gssr-record -o test-report-00 sleep 5 && cat test-report-00/step_0/proc_0.{csv,meta.txt}
./gssr-record -o test-report-00 -- sleep 5 && cat test-report-00/step_0/proc_0.{csv,meta.txt}

srun -N1 -n1 -t 00:01:00 ./gssr-record -o test-report-02 /usr/bin/dcgmproftester12 -t 1006 -d 240 
./gssr-analyze.py -i test-report-02 -o test-report-02.pdf

if false; then

srun -N1 -n1 --signal=HUP@30 -t 00:01:00 ./gssr-record -o test-report-03 /usr/bin/dcgmproftester12 -t 1006 -d 240 
./gssr-analyze.py -i test-report-03 -o test-report-03.pdf

srun -N3 -n3 -t 00:30:00 ./gssr-record -o test-report-04 /usr/bin/dcgmproftester12 -t 1006 -d 3600 
./gssr-analyze.py -i test-report-04 -o test-report-04.pdf


#time ./gssr-record -o test-report-00 -- sleep 300

rm -rf dump_evrard.h5 dump_sedov.h5 test-report-05
OMP_NUM_THREADS=12 time srun -N3 -n3 -c12 -t 00:03:00 --gpus-per-task=1 ./gssr-record -o test-report-05 sphexa/build/main/src/sphexa/sphexa-cuda --init evrard --glass 50c.h5 -n 200 -s 2000000 -w 2000000
./gssr-analyze.py -i test-report-05 -o test-report-05.pdf

fi

cat > test-ubuntu.toml <<EOF
image = "library/ubuntu:24.04"
mounts = ["${SCRATCH}:${SCRATCH}", "${HOME}:${HOME}"]
workdir = "${SCRATCH}"
EOF
srun --environment=./test-ubuntu.toml $(realpath ./gssr-record) --help

cat > test-ubuntu.toml <<EOF
image = "library/ubuntu:24.04"
mounts = ["${SCRATCH}:${SCRATCH}", "${HOME}:${HOME}"]
workdir = "${SCRATCH}"

[annotations]
com.hooks.dcgm.enabled = "true"
EOF
srun --environment=./test-ubuntu.toml $(realpath ./gssr-record) --help


exit
srun -N2 -n6 ./gssr-record -o test-report-01 /usr/bin/dcgmproftester12 -t 1006 -d 20 \
    && ./gssr-analyze.py -i test-report-01 -o test-report-01.pdf
