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

A simple example to test the setup is to record a few seconds of no activity.
```bash
./gssr-record -o gr-sleep-test sleep 30
./gssr-analyze.py gr-sleep-test -o gr-sleep-test-report.pdf
```
