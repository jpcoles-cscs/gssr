# gssr
A new simplified version of the GPU saturation scorer for CSCS workloads

# Getting Started
The analysis tool depends on uv to create a python environment with the
necessary dependencies. If uv is not installed, run
```bash
make install-uv
```

To create the gssr-record executable run
```bash
make
```

# Recording Metrics
```bash
Usage: gssr-record [OPTIONS] <cmd> [args...]
Run cmd and record GPU metrics. Results can be given to the
GPU saturation scorer (GSSR) to produce a report for CSCS project proposals.

   -h | --help         Display this help message.
   --version           Show version information.
   -o <directory>      Create directory and write results there.

gssr-record depends on the NVIDIA DCGM library. When running in a
container at CSCS you will need the Container Engine annotation in
your container EDF file:

[annotations]
com.hooks.dcgm.enabled = "true"
```

# Generating a report
```bash
usage: gssr-analyze.py [-h] [-o OUTPUT] [directory ...]

Generate a report from GPU metrics collected with gssr-record.

positional arguments:
  directory             Top level directories created by gssr-record

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output PDF filename (default: gssr-report.pdf)
```

# Quick Start Example
A simple example to test the setup is to record a few seconds of no activity.
```bash
./gssr-record -o gr-sleep-test sleep 30
./gssr-analyze.py gr-sleep-test -o gr-sleep-test-report.pdf
```
