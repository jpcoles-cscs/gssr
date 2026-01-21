#!/bin/bash

ROOT=generated-report

function one_step()
{
    stepname=$1
    mkdir -p ${ROOT}/${stepname}

    OUTMETA=${ROOT}/${stepname}/proc_0.meta.txt
    OUTDATA=${ROOT}/${stepname}/proc_0.csv

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

    python3 ./fake-csv.py --seconds 100 --gpus 4 --output $OUTDATA

}

one_step step_0
