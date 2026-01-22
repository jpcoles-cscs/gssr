#!/bin/bash

export SLURM_PARTITION=debug

make clean
make
make install-uv

mkdir testing-tmp
cd testing-tmp

GA=$(realpath ../gssr-analyze.py)
GR=$(realpath ../gssr-record)

set -x
#set -e

make -f ../Makefile clean-tests

function test_gr_basic()
{
    $GR -h
    $GR --help
    $GR --version
    $GR --help -o test-report-00
}

function test_ga_basic()
{
    $GA DOES_NOT_EXIST
    $GA test-report-01
    stat report.pdf

    mkdir -p test-report-01/step_0
    $GA test-report-01

    touch test-report-01/step_0/proc_0.meta.txt
    $GA test-report-01
}

function test_00_sleep()
{
    $GR -o -- sleep
    $GR -o test-report-00 sleep 5 && cat test-report-00/step_0/proc_0.{csv,meta.txt}
    $GR -o test-report-00 -- sleep 5 && cat test-report-00/step_0/proc_0.{csv,meta.txt}
}

function test_00_dir_permission()
{
    $GR -o $SCRATCH/gssr-test -- sleep 1
    ls -ld $SCRATCH/gssr-test
}

function test_01_dcgmproftester()
{
    srun -N2 -n6 $GR -o test-report-01 /usr/bin/dcgmproftester12 -t 1006 -d 20 \
        && $GA test-report-01 -o test-report-01.pdf
}

function test_02_dcgmproftester()
{
    srun -N1 -n1 -t 00:01:00 $GR -o test-report-02a /usr/bin/dcgmproftester12 -t 1006 -d 240 
    $GA test-report-02a -o test-report-02a.pdf

    srun -N1 -n4 --gpus-per-task=1 -t 00:01:00 $GR -o test-report-02b /usr/bin/dcgmproftester12 -t 1006 -d 240 
    $GA test-report-02b -o test-report-02b.pdf

    srun -N32 --ntasks-per-node=4 --gpus-per-task=1 -t 00:10:00 $GR -o test-report-02c /usr/bin/dcgmproftester12 -t 1006 -d 600  --max-processes 1
    $GA test-report-02c -o test-report-02c.pdf
}

function test_03_signal()
{
    srun -N1 -n1 --signal=HUP@30 -t 00:01:00 $GR -o test-report-03 /usr/bin/dcgmproftester12 -t 1006 -d 240
    $GA test-report-03 -o test-report-03.pdf
}

function test_04_long_running()
{
    srun -N3 -n3 -t 00:30:00 $GR -o test-report-04 /usr/bin/dcgmproftester12 -t 1006 -d 3600 
    $GA test-report-04 -o test-report-04.pdf
}

function test_05_sphexa()
{
    rm -rf dump_evrard.h5 dump_sedov.h5 test-report-05
    OMP_NUM_THREADS=12 time srun -N3 -n3 -c12 -t 00:03:00 --gpus-per-task=1 $GR -o test-report-05 sphexa/build/main/src/sphexa/sphexa-cuda --init evrard --glass 50c.h5 -n 200 -s 2000000 -w 2000000
    $GA test-report-05 -o test-report-05.pdf
}

function test_06_mps_wrapper()
{
    srun -N1 -n32 -t 00:05:00 ./mps-wrapper.sh $GR -o test-report-06 /usr/bin/dcgmproftester12 -t 1006 -d 60 
    $GA test-report-06 -o test-report-06.pdf
}

function test_07_multi_mps_wrapper()
{
    cat > test-report-07.sh <<EOF
#!/bin/bash
#SBATCH -J test-report-07
#SBATCH -t 05:00
#SBATCH -N1
#SBATCH -n32
#SBATCH -A csstaff
#SBATCH -o test-report-07.slurm.out

srun ./mps-wrapper.sh $GR -o test-report-07 /usr/bin/dcgmproftester12 -t 1006 -d 60 
srun ./mps-wrapper.sh $GR -o test-report-07 /usr/bin/dcgmproftester12 -t 1007 -d 120 
EOF

    sbatch -W test-report-07.sh
    $GA test-report-07 -o test-report-07.pdf
}

function test_container()
{
    cat > test-ubuntu.toml <<EOF
    image = "library/ubuntu:24.04"
    mounts = ["${SCRATCH}:${SCRATCH}", "${HOME}:${HOME}"]
    workdir = "${SCRATCH}"
EOF
    srun --environment=./test-ubuntu.toml $(realpath $GR) --help

    cat > test-ubuntu.toml <<EOF
    image = "library/ubuntu:24.04"
    mounts = ["${SCRATCH}:${SCRATCH}", "${HOME}:${HOME}"]
    workdir = "${SCRATCH}"

    [annotations]
    com.hooks.dcgm.enabled = "true"
EOF
    srun --environment=./test-ubuntu.toml $(realpath $GR) --help

}

function test_08_concurrent_srun()
{
    cat > test-report-08.sh <<EOF
#!/bin/bash
#SBATCH -J test-report-08
#SBATCH -t 05:00
#SBATCH -N1
# BATCH -n4
#SBATCH -A csstaff
# BATCH --gpus-per-node=4
#SBATCH -o test-report-08.slurm.out
#SBATCH --exclusive --mem=450G


srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=5 --mem=50G  $GR -o test-report-08 /usr/bin/dcgmproftester12 --max-processes 1 -t 1006 -d 20 &
srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=5 --mem=50G  $GR -o test-report-08 /usr/bin/dcgmproftester12 --max-processes 1 -t 1007 -d 20 &
srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=5 --mem=50G  $GR -o test-report-08 /usr/bin/dcgmproftester12 --max-processes 1 -t 1008 -d 20 &
srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=5 --mem=50G  $GR -o test-report-08 /usr/bin/dcgmproftester12 --max-processes 1 -t 1005 -d 20 &

wait
EOF

    sbatch -W test-report-08.sh
    $GA test-report-08 -o test-report-08.pdf
}

test_gr_basic
test_ga_basic
test_00_sleep
test_01_dcgmproftester
test_02_dcgmproftester
test_03_signal
test_04_long_running
test_05_sphexa
test_06_mps_wrapper
test_07_multi_mps_wrapper
test_container
test_08_concurrent_srun
test_00_dir_permission
