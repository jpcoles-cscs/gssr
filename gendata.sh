#!/bin/bash

ROOT=testing-tmp/

function one_step()
{
    dir=${ROOT}${1}
    stepname=$2
    proc=$3
    mkdir -p ${dir}/${stepname}

    OUTMETA=${dir}/${stepname}/${proc}.meta.txt
    OUTDATA=${dir}/${stepname}/${proc}.csv

    if ! stat ${dir}/${stepname}/proc_0.meta.txt; then

    cat > $OUTMETA <<EOF
{
    "gssr-record-version": "2.0",
    "date": "2026-01-20T15:38:10+0100",
    "cluster": "alps-daint",
    "jobid": "12345678",
    "jobname": "fake-job",
    "nnodes": "1",
    "ntasks": "1",
    "ngpus": "1",
    "step_nnodes": "1",
    "step_ntasks": "1",
    "executable": "my-fake-data-generator",
    "arguments": ""
}

EOF

    fi

    
    #python3 ./fake-csv.py --seconds $4 --gpus $5 --output $OUTDATA
}

#one_step fake_00 step_0 proc_0 100 0
#one_step fake_00 step_0 proc_3 100 0
#./gssr-analyze.py ${ROOT}fake_00 -o fake-report-00.pdf

# one_step fake_01 step_0 proc_0 100 4
# one_step fake_01 step_0 proc_3 100 4
#./gssr-analyze.py ${ROOT}fake_01 -o fake-report-01.pdf

# Check missing meta file
rsync -a ${ROOT}fake_01/ ${ROOT}fake_01a
rm ${ROOT}fake_01a/step_0/proc_0.meta.txt
./gssr-analyze.py ${ROOT}fake_01a -o fake-report-01a.pdf
  
# Check missing data files
rsync -a ${ROOT}fake_01/ ${ROOT}fake_01b
rm ${ROOT}fake_01b/step_0/proc_*.csv
./gssr-analyze.py ${ROOT}fake_01b -o fake-report-01b.pdf

# Check missing meta and data files
rsync -a ${ROOT}fake_01/ ${ROOT}fake_01c
rm ${ROOT}fake_01c/step_0/proc_*.csv
rm ${ROOT}fake_01c/step_0/proc_*.meta.txt
./gssr-analyze.py ${ROOT}fake_01c -o fake-report-01c.pdf

# Check missing step_dir
rsync -a ${ROOT}fake_01/ ${ROOT}fake_01d
(cd ${ROOT}fake_01d && rm -rf step_0)
./gssr-analyze.py ${ROOT}fake_01d -o fake-report-01d.pdf

# one_step fake_02 step_0 proc_0 3000 4
# one_step fake_02 step_0 proc_3 3000 4
./gssr-analyze.py ${ROOT}fake_02 -o fake-report-02.pdf
# 
# one_step fake_03 step_0 proc_0 3000 10
# one_step fake_03 step_0 proc_3 3000 10
# ./gssr-analyze.py ${ROOT}fake_03 -o fake-report-03.pdf
# 
# one_step fake_04 step_0 proc_0 3000 10
# one_step fake_04 step_0 proc_3 30000 5
# ./gssr-analyze.py ${ROOT}fake_04 -o fake-report-04.pdf
# 
# for i in $(seq 0 1000); do
#     one_step fake_05 step_0 proc_$i 3000 10
# done
#./gssr-analyze.py ${ROOT}fake_05 -o fake-report-05.pdf
