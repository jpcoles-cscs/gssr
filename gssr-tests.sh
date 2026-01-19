#!/bin/bash

set -x
#set -e

make clean
make clean-tests
make
make install-uv

./gssr-record -h
./gssr-record --help
./gssr-record --version
./gssr-record --help -o test-report-00

./gssr-analyze.py DOES_NOT_EXIST
./gssr-analyze.py test-report-01
stat report.pdf

#./gssr-record -o -- sleep
#./gssr-record -o test-report-00 sleep 5 && cat test-report-00/step_0/proc_0.{csv,meta.txt}
#./gssr-record -o test-report-00 -- sleep 5 && cat test-report-00/step_0/proc_0.{csv,meta.txt}
#

#srun -N1 -n32 -t 00:05:00 ./mps-wrapper.sh ./gssr-record -o test-report-06 /usr/bin/dcgmproftester12 -t 1006 -d 60 
#./gssr-analyze.py test-report-06 -o test-report-06.pdf

cat > test-report-07.sh <<EOF
#!/bin/bash
#SBATCH -J test-report-07
#SBATCH -t 05:00
#SBATCH -N1
#SBATCH -n32
#SBATCH -A csstaff
#SBATCH -o test-report-07.slurm.out

srun ./mps-wrapper.sh ./gssr-record -o test-report-07 /usr/bin/dcgmproftester12 -t 1006 -d 60 
srun ./mps-wrapper.sh ./gssr-record -o test-report-07 /usr/bin/dcgmproftester12 -t 1007 -d 120 
EOF

sbatch -W test-report-07.sh
./gssr-analyze.py test-report-07 -o test-report-07.pdf

exit

#srun -N1 -n1 -t 00:01:00 ./gssr-record -o test-report-02a /usr/bin/dcgmproftester12 -t 1006 -d 240 
#./gssr-analyze.py test-report-02a -o test-report-02a.pdf

#srun -N1 -n4 --gpus-per-task=1 -t 00:01:00 ./gssr-record -o test-report-02b /usr/bin/dcgmproftester12 -t 1006 -d 240 
#./gssr-analyze.py test-report-02b -o test-report-02b.pdf

srun -N32 --ntasks-per-node=4 --gpus-per-task=1 -t 00:10:00 ./gssr-record -o test-report-02c /usr/bin/dcgmproftester12 -t 1006 -d 600  --max-processes 1
./gssr-analyze.py test-report-02c -o test-report-02c.pdf

exit

if false; then

srun -N1 -n1 --signal=HUP@30 -t 00:01:00 ./gssr-record -o test-report-03 /usr/bin/dcgmproftester12 -t 1006 -d 240
./gssr-analyze.py test-report-03 -o test-report-03.pdf

srun -N3 -n3 -t 00:30:00 ./gssr-record -o test-report-04 /usr/bin/dcgmproftester12 -t 1006 -d 3600 
./gssr-analyze.py test-report-04 -o test-report-04.pdf


#time ./gssr-record -o test-report-00 -- sleep 300

rm -rf dump_evrard.h5 dump_sedov.h5 test-report-05
OMP_NUM_THREADS=12 time srun -N3 -n3 -c12 -t 00:03:00 --gpus-per-task=1 ./gssr-record -o test-report-05 sphexa/build/main/src/sphexa/sphexa-cuda --init evrard --glass 50c.h5 -n 200 -s 2000000 -w 2000000
./gssr-analyze.py test-report-05 -o test-report-05.pdf

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
    && ./gssr-analyze.py test-report-01 -o test-report-01.pdf
