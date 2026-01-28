#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.6"
# dependencies = [
#   "numpy",
#   "matplotlib",
#   "pandas",
# ]
# ///

#
# Generate a PDF file from GPU metrics recorded by gssr-record.
#
# Run gssr-analyze --help for usage.
#
# Written by Jonathan Coles <jonathan.coles@cscs.ch>
#

import sys,os
import argparse
import json
import textwrap
import copy
import glob
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator


# Random Number Generator used to add a little noise to the histrogram
# to increase legibility when bars overlap.
rng = np.random.default_rng()


def load_metrics_and_meta(paths): 
    """
    Walk a directory tree and load all CSV files into a single DataFrame.

    Parameters
    ----------
    paths : list
        List of directories to look for metric data

    Returns
    -------
    pandas.DataFrame, pandas.DataFrame
    """
    frames = []
    meta = []

    for path in paths:

        steps = glob.glob(os.path.join(path, 'step_*'))

        for step in steps:

            meta_files = glob.glob(os.path.join(step, 'proc_*.meta.txt'))
            proc_files = glob.glob(os.path.join(step, 'proc_*.csv'))

            if len(meta_files) > 1:
                print(f'Too many metadata files in {step}. Skipping.')
                continue

            if len(meta_files) == 0:
                print(f'Missing metadata file in {step}. Attempting to continue.')
                metadf = dict()
            else:
                meta_file = meta_files[0]

                m = re.search(r"step_(\d+)/proc_(\d+)\.meta\.txt$", meta_file)
                istep, iproc = int(m.group(1)), int(m.group(2))

                try:
                    with open(meta_file) as f:
                        metadf = json.load(f)
                except json.decoder.JSONDecodeError as e:
                    print(f'Error reading {meta_file}. Attempting to continue. {e}')
                    metadf = dict()

            m = re.search(r"step_(\d+)$", step)
            istep = int(m.group(1))

            ngpus  = 0
            energy = 0
            appended_frames = 0
            for proc_file in proc_files:

                m = re.search(r"step_(\d+)/proc_(\d+)\.csv$", proc_file)
                istep, iproc = int(m.group(1)), int(m.group(2))

                try:
                    df = pd.read_csv(proc_file)

                    # Use filename (without extension) as label
                    df['report'] = path
                    df['step'] = istep
                    df['proc'] = iproc

                    # Count the number of unique GPUs this step monitored
                    ngpus  += df['gpuId'].nunique()
                    energy += df['DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION_avg'].sum()

                    frames.append(df)
                    appended_frames += 1
                except pd.errors.EmptyDataError:
                    pass

            if appended_frames == 0:
                print(f'No metric data found for {step}. Attempting to continue.')

            metadf['report'] = [path]
            metadf['step'] = [istep]
            metadf['unique gpus'] = [ngpus]
            metadf['gpu energy (mJ)'] = [energy]
            meta.append(pd.DataFrame(metadf))

    df     = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    metadf = pd.concat(meta,   ignore_index=True) if meta   else pd.DataFrame()

    return df, metadf

def plot_memory_metrics(ax, df):
    """
    Plot the GPU memory usage metrics.

    Parameters
    ----------
    ax : np.array
        1x2 array of Axes objects
    df : pandas.DataFrame
        DataFrame containing all metrics

    Returns
    -------
    None
    """


    cfg = [
        ['DCGM_FI_DEV_FB_FREE_avg',     1e-3, 'g',        'Free'], 
        ['DCGM_FI_DEV_FB_USED_avg',     1e-3, 'r',        'Used'],
        ['DCGM_FI_DEV_FB_RESERVED_avg', 1e-3, 'orange',   'Reserved']]

    x = df['timestamp']

    for metric,scale,c,_ in cfg:
        y_avg = df[metric,'mean'] * scale
        min_y = df[metric,'min'] * scale
        max_y = df[metric,'max'] * scale
    
        #figpath = self.plot_time_series(t, y_avg, min_y, max_y, metric)

        # Downsample data to a maximum of 100 points
        #x, y_avg, min_y, max_y = self.downsample((x, y_avg, min_y, max_y))
        
        # Add the shaded area between min_y and max_y
        #ax1.fill_between(x, min_y, max_y, color='lightblue', alpha=0.5, label='Range')
        
        # Plot the actual y line over the shaded area
        ax[0].plot(x, y_avg, label=metric, color=c, linewidth=1.0)
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel(f'GPU Memory Usage (GB)', labelpad=10)

        n_bins = 20 
        y,bins = np.histogram(y_avg, weights=np.full_like(y_avg, 1./len(y_avg)) * 100, bins=n_bins, range=(0,100))
        if np.amax(y) < 20:
            xeps, yeps = 0,0
        else:
            yeps = rng.integers(low=0, high=2, size=len(y))
            yeps[y <= 0] = 0
            xeps = rng.integers(low=0, high=2, size=len(y))
        ax[1].plot(y + yeps, bins[0:-1] + xeps, alpha=0.8, color=c, lw=1.0, drawstyle='steps')

    ax[1].set_title('Histogram', fontsize=6)
    ax[1].set_xlabel('% of Runtime')
    ax[1].set_xticks([0,25,50,75,100])
    ax[1].set_xlim(-5, 105)
    ax[1].set_ylim(ax[0].get_ylim())

    legend = [ [legend_text,   pl.Line2D([0], [0], lw=5, color=c)] for _,scale,c,legend_text in cfg ]
    l,h = list(zip(*legend))
    ax[0].legend(h,l, ncol=len(legend), loc='lower left', borderpad=0, **dict(frameon=False, bbox_to_anchor=(0.0, 1.0), fontsize=6))

def plot_txrx_metrics(ax, df):
    """
    Plot the GPU data transmission metrics like PCIe and NVlink

    Parameters
    ----------
    ax : np.array
        1x2 array of Axes objects
    df : pandas.DataFrame
        DataFrame containing all metrics

    Returns
    -------
    None
    """

    cfg = [
        ['DCGM_FI_PROF_PCIE_TX_BYTES_avg',      1e-6, 'limegreen',        'PCIe Send'], 
        ['DCGM_FI_PROF_PCIE_RX_BYTES_avg',      1e-6, 'darkgreen',        'PCIe Recv'],
        ['DCGM_FI_PROF_NVLINK_TX_BYTES_avg',    1e-6, 'b',        'NVLink Send'],
        ['DCGM_FI_PROF_NVLINK_RX_BYTES_avg',    1e-6, 'darkblue',   'NVLink Recv'],
        ['DCGM_FI_PROF_C2C_TX_ALL_BYTES',       1e-6, 'orange',        'C2C Send'],
        ['DCGM_FI_PROF_C2C_RX_ALL_BYTES',       1e-6, 'darkorange',   'C2C Recv']]
    cfg = [c for c in cfg if c[0] in df.columns]

    x = df['timestamp']

    data_range = [np.inf, -np.inf]
    for metric,scale,c,_ in cfg:
        y_avg = df[metric,'mean'] * scale
    
        # Plot the actual y line over the shaded area
        ax[0].plot(x, y_avg, color=c, linewidth=1.0)
        data_range = [min(data_range[0], np.amin(y_avg)), max(data_range[1], np.amax(y_avg))]

    for metric,scale,c,_ in cfg:
        y_avg = df[metric,'mean'] * scale

        n_bins = 20 
        y,bins = np.histogram(y_avg, weights=np.full_like(y_avg, 1./len(y_avg)) * 100, bins=n_bins, range=data_range)
        if np.amax(y) < 20:
            xeps, yeps = 0,0
        else:
            yeps = rng.integers(low=0, high=2, size=len(y))
            yeps[y <= 0] = 0
            xeps = rng.integers(low=0, high=2, size=len(y))
        ax[1].plot(y, bins[0:-1], alpha=0.8, color=c, lw=1.0, drawstyle='steps')

    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel(f'Data Movement (MB/s)', labelpad=10)
    ax[0].set_ylim(ymin=-5, ymax=max(ax[0].get_ylim()[1]+5, 50))

    ax[1].set_title('Histogram', fontsize=6)
    ax[1].set_xlabel('% of Runtime')
    ax[1].set_xticks([0,25,50,75,100])
    ax[1].set_xlim(-5, 105)
    ax[1].set_ylim(ax[0].get_ylim())
    
    legend = [ [legend_text,   pl.Line2D([0], [0], lw=5, color=c)] for _,_,c,legend_text in cfg ]
    l,h = list(zip(*legend))
    ax[0].legend(h,l, ncol=len(legend), loc='lower left', borderpad=0, **dict(frameon=False, bbox_to_anchor=(0.0, 1.0), fontsize=6))

def plot_active_metrics(ax, df):
    """
    Plot the GPU utilization and Tensor/FP metrics.

    Parameters
    ----------
    ax : np.array
        1x2 array of Axes objects
    df : pandas.DataFrame
        DataFrame containing all metrics

    Returns
    -------
    None
    """

    cfg = [
        ['DCGM_FI_DEV_GPU_UTIL_avg',              1, 'k', 'GPU Util'], 
        ['DCGM_FI_PROF_PIPE_TENSOR_ACTIVE_avg', 100, 'm', 'Tensor'], 
        ['DCGM_FI_PROF_PIPE_FP64_ACTIVE_avg',   100, 'r', 'fp64'],
        ['DCGM_FI_PROF_PIPE_FP32_ACTIVE_avg',   100, 'g', 'fp32'],
        ['DCGM_FI_PROF_PIPE_FP16_ACTIVE_avg',   100, 'b', 'fp16'],
        ['DCGM_FI_PROF_DRAM_ACTIVE_avg',        100, 'orange', 'DRAM']
        ]

    x = df['timestamp']

    for metric,scale,c,_ in cfg:
        y_avg = df[metric,'mean'] * scale
        min_y = df[metric,'min'] * scale
        max_y = df[metric,'max'] * scale

        # Downsample data to a maximum of 100 points
        #x, y_avg, min_y, max_y = self.downsample((x, y_avg, min_y, max_y))
        
        
        # Add the shaded area between min_y and max_y
        #ax1.fill_between(x, min_y, max_y, color='lightblue', alpha=0.5, label='Range')
        
        # Plot the actual y line over the shaded area
        ax[0].plot(x, y_avg, label=metric, color=c, linewidth=1.0)

        # Create the distribution plot (right)
        n_bins = 20 #min(100, max(10, int(np.ceil(np.sqrt(len(y_avg))))))
        y,bins = np.histogram(y_avg, weights=np.full_like(y_avg, 1./len(y_avg)) * 100, bins=n_bins, range=(0,100))

        if np.amax(y) < 20:
            xeps, yeps = 0,0
        else:
            yeps = rng.integers(low=0, high=2, size=len(y))
            yeps[y <= 0] = 0
            xeps = rng.integers(low=0, high=2, size=len(y))
        ax[1].plot(y + yeps, bins[0:-1] + xeps, alpha=0.8, color=c, lw=1.0, drawstyle='steps')

    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel(f'GPU activity (%)', labelpad=10)
    ax[0].set_ylim(ymin=-5, ymax=105)

    ax[1].set_title('Histogram', fontsize=6)
    ax[1].set_xlabel('% of Runtime')
    ax[1].set_xticks([0,25,50,75,100])
    ax[1].set_xlim(-5, 105)
    ax[1].set_ylim(ax[0].get_ylim())
    

    legend = [ [legend_text,   pl.Line2D([0], [0], lw=5, color=c)] for _,_,c,legend_text in cfg ]
    l,h = list(zip(*legend))
    ax[0].legend(h,l, ncol=len(legend), loc='lower left', borderpad=0, **dict(frameon=False, bbox_to_anchor=(0.0, 1.0), fontsize=6))

def plot_energy_metrics(ax, df):
    """
    Plot the GPU energy and power usage.

    Parameters
    ----------
    ax : np.array
        1x2 array of Axes objects
    df : pandas.DataFrame
        DataFrame containing all metrics

    Returns
    -------
    None
    """

    cfg = [
        ['DCGM_FI_DEV_POWER_USAGE_avg',                 1, 'red', 'GPU Power']
        ]

    x = df['timestamp']

    metric,scale,c,_ = cfg[0]
    y_avg = df[metric,'mean'] * scale

    ax[0].plot(x, y_avg, label=metric, color=c, linewidth=1.0)
    data_range = [np.amin(y_avg), np.amax(y_avg)]


    # Create the distribution plot (right)
    n_bins = 20 #min(100, max(10, int(np.ceil(np.sqrt(len(y_avg))))))
    y,bins = np.histogram(y_avg, weights=np.full_like(y_avg, 1./len(y_avg)) * 100, bins=n_bins, range=data_range)

    if np.amax(y) < 20:
        xeps, yeps = 0,0
    else:
        yeps = rng.integers(low=0, high=2, size=len(y))
        yeps[y <= 0] = 0
        xeps = rng.integers(low=0, high=2, size=len(y))
    ax[1].plot(y + yeps, bins[0:-1] + xeps, alpha=0.8, color=c, lw=1.0, drawstyle='steps')
    ax[1].tick_params(labelright=False)

    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel(f'Power (W)', labelpad=10)
    ax[0].set_ylim(ymin=-5, ymax=max(ax[0].get_ylim()[1]+5, 50))

    ax[1].set_title('Histogram', fontsize=6)
    ax[1].set_xlabel('% of Runtime')
    ax[1].set_xticks([0,25,50,75,100])
    ax[1].set_xlim(-5, 105)
    ax[1].set_ylim(ax[0].get_ylim())

    legend = [ [legend_text,   pl.Line2D([0], [0], lw=5, color=c)] for _,_,c,legend_text in cfg ]
    l,h = list(zip(*legend))
    ax[0].legend(h,l, ncol=len(legend), loc='lower left', borderpad=0, **dict(frameon=False, bbox_to_anchor=(0.0, 1.0), fontsize=6))


def plot_sm_metrics(ax, df):
    """
    Plot the GPU SM active and occupancy metrics.

    Parameters
    ----------
    ax : np.array
        1x2 array of Axes objects
    df : pandas.DataFrame
        DataFrame containing all metrics

    Returns
    -------
    None
    """

    cfg = [
        ['DCGM_FI_PROF_SM_ACTIVE_avg',      100, 'g', 'SM Active'], 
        ['DCGM_FI_PROF_SM_OCCUPANCY_avg',   100, 'r', 'SM Occupancy']]

    x = df['timestamp']

    for metric,scale, c,_ in cfg:
        y_avg = df[metric,'mean'] * scale
        min_y = df[metric,'min'] * scale
        max_y = df[metric,'mean'] * scale
    
#       if isMetricRatio(metric) :
#           y_avg = y_avg*100
#           min_y = min_y*100
#           max_y = max_y*100

        # Downsample data to a maximum of 100 points
        #x, y_avg, min_y, max_y = self.downsample((x, y_avg, min_y, max_y))
        
        # Add the shaded area between min_y and max_y
        #ax1.fill_between(x, min_y, max_y, color='lightblue', alpha=0.5, label='Range')
        
        # Plot the actual y line over the shaded area
        ax[0].plot(x, y_avg, label=metric, color=c, linewidth=1.0)
        #ax1.grid(alpha=0.8)

        # Create the distribution plot (right)
        n_bins = 20 
        y,bins = np.histogram(y_avg, weights=np.full_like(y_avg, 1./len(y_avg)) * 100, bins=n_bins, range=(0,100))
        yeps = rng.integers(low=0, high=2, size=len(y))
        yeps[y <= 0] = 0
        xeps = rng.integers(low=0, high=2, size=len(y))
        ax[1].plot(y + yeps, bins[0:-1] + xeps, alpha=0.8, color=c, lw=1.0, drawstyle='steps')

    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel(f'SM Usage (%)', labelpad=10)
    ax[0].set_ylim(ymin=-5, ymax=105)
    
    ax[1].set_title('Histogram', fontsize=6)
    ax[1].set_xlabel('% of Runtime')
    ax[1].set_xticks([0,25,50,75,100])
    ax[1].set_xlim(-5, 105)
    ax[1].set_ylim(ax[0].get_ylim())

    
    legend = [ [legend_text,   pl.Line2D([0], [0], lw=5, color=c)] for _,_,c,legend_text in cfg ]
    l,h = list(zip(*legend))
    ax[0].legend(h,l, ncol=len(legend), loc='lower left', borderpad=0, **dict(frameon=False, bbox_to_anchor=(0.0, 1.0), fontsize=6))

def title_page_table(ax, metadf):
    """
    Draw a page on the title page containing job metadata.

    Parameters
    ----------
    ax : Axes
        A single axis covering the A4 page.
    metadf : pandas.DataFrame
        DataFrame containing job metadata.

    Returns
    -------
    None
    """

    wrap_width = 45  # wrap lines at 30 characters

    if metadf is None:
        # Create table
        table = ax.table(
            cellText=['No data found'],
            cellLoc='center',
            loc='top',
            bbox = (0,0,1,0.5),
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        
        return

    m = metadf.iloc[0].to_dict()

    energy = m.get('gpu energy (mJ)', 0) * 1e-3 * 0.000277778
    table_data = [
        [ 'Slurm Job ID',           m.get('jobid',        'missing')  ],
        [ 'Job Name',               m.get('jobname',      'missing')  ],
        [ 'Jobstep',                m.get('step',         'missing')  ],
        [ 'Cluster',                m.get('cluster',      'missing')  ],
        [ 'Date',                   m.get('date',         'missing')  ],
        [ 'Node Count',             '%s (of %s)' % (m.get('step_nnodes',  'missing'), m.get('nnodes',  'unknown'))  ],
        [ 'Task Count',             '%s (of %s)' % (m.get('step_ntasks',  'missing'), m.get('ntasks',  'unknown'))  ],
        [ 'GPU Count',              m.get('unique gpus',  'missing')  ],
        [ 'GPU Energy Use',         ('%.2f Wh' % energy) if energy > 0 else 'missing'  ],
        [ 'Executable',             m.get('executable',   'missing')  ],
        [ 'Arguments',              m.get('arguments',    'missing')  ],
    ]

    row_line_counts = []  # to store number of lines per row
    for row in table_data:
        label, val = row
        val_str = str(val)
        wrapped_val = textwrap.wrap(val_str, width=wrap_width, max_lines=15)
        row[1] = '\n'.join(wrapped_val)
        row_line_counts.append(len(wrapped_val))

    # Create table
    table = ax.table(
        cellText=table_data,
        #colLabels=['Field', 'Value'],
        cellLoc='left',
        loc='top',
        bbox = (0,0,1,0.55),
        colWidths = [0.3, 0.7]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for i, lines in enumerate(row_line_counts):  # header row = 1 line
        lines = max(1, lines)
        table[i, 0].set_height(0.04 * lines)
        table[i, 1].set_height(0.04 * lines)

    # --- Apply alternating row colors ---
    colors = ['#ffffff', '#dddddd']  # white and light grey
    for i in range(0, len(row_line_counts)):  # skip header row
        color = colors[(i-1) % 2]  # alternate colors
        table[i, 0].set_facecolor(color)
        table[i, 1].set_facecolor(color)


def title_page(pdf, df, metadf):
    """
    Draw the title page.

    Parameters
    ----------
    pdf : 
        The open pdf file handle to save the title page figure into.
    df : pandas.DataFrame
        DataFrame containing job metrics.
    metadf : pandas.DataFrame
        DataFrame containing job metadata.

    Returns
    -------
    None
    """

    fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(8, 11), squeeze=False)
    ax = axes[0,0]

    ax.axis('off')  # hide axes

    fig.text(
        0.5, 0.95,
        'CSCS Job Report',
        ha='center',
        va='top',
        fontsize=24,
        fontweight='bold'
    )

    from datetime import datetime
    # Current timestamp in YYYY-MM-DD HH:MM:SS format
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    timestamp_text = f'Generated on {current_time}'

    # Smaller, italic, centered subtitle below title
    fig.text(
        0.5, 0.91,          # slightly below the title
        timestamp_text,
        ha='center',
        va='top',
        fontsize=12,
        fontstyle='italic',
        color='gray'
    )

    if df is None:
        fig.text(
            0.5, 0.7,
            'This jobstep has no data',
            ha='center',
            va='center',
            fontsize=18,
            fontweight='bold',
            c='red'
        )

    title_page_table(ax, metadf)

    pdf.savefig(fig)
    pl.close(fig)


def definitions_page(pdf):
    """
    Make a table containing the definition of terms used in the report.

    Parameters
    ----------
    pdf : 
        The open pdf file handle to save the title page figure into.

    Returns
    -------
    None
    """

    fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(8, 11), squeeze=False)
    ax = axes[0,0]

    ax.axis('off')  # hide axes

    fig.text(
        0.1, 0.95,
        'Definitions',
        ha='left',
        va='top',
        fontsize=16,
        fontweight='bold'
    )

    fig.text(
        0.1, 0.93,
        'From Nvidia Documentation',
        ha='left',
        va='top',
        fontsize=12,
        fontstyle='italic',
        color='gray'
    )

    col0_width = 25
    col1_width = 55

    table_data = [
        [ 'GPU Memory (MB) Used/Free/Reserved',
             'Used/Free/Reserved Frame Buffer of the GPU in MB' ],
        [ 'GPU Utilization (%)',
             ('The fraction of time any portion of the graphics or compute engines'
             ' were active. The graphics engine is active if a graphics/compute'
             ' context is bound and the graphics/compute pipe is busy. The value'
             ' represents an average over a time interval and is not an instantaneous'
             ' value.')],
        [ 'SM Active (%)',
             'The fraction of time at least one warp was active on a' +
             ' multiprocessor, averaged over all multiprocessors. Note that "active"' +
             ' does not necessarily mean a warp is actively computing.' +
             ' Warps waiting on memory requests are considered active. The value' +
             ' represents an average over a time interval and is not an instantaneous' +
             ' value. '],
        [ 'SM Occupancy (%)',
             'The fraction of resident warps on a multiprocessor, relative to the' +
             'maximum number of concurrent warps supported on a multiprocessor. The' +
             'value represents an average over a time interval and is not an' +
             'instantaneous value. Higher occupancy does not necessarily indicate' +
             'better GPU usage. For GPU memory bandwidth limited workloads, higher' +
             'occupancy is indicative of more effective GPU usage. However if the' +
             'workload is compute limited, then higher occupancy does not' +
             'necessarily correlate with more effective GPU usage.'],

        [ 'Tensor Active (%)',
             'The fraction of cycles the tensor (HMMA / IMMA) pipe was active. The' +
             'value represents an average over a time interval and is not an' +
             'instantaneous value. Higher values indicate higher utilization of the' +
             'Tensor Cores. An activity of 100% is equivalent to issuing a' +
             'tensor instruction every other cycle for the entire time interval.'],

        [ 'FP Active (%) for FP64, FP32, FP16',
             'The fraction of cycles the respective float-point pipe was' +
             'active. The value represents an average over a time interval and is' +
             'not an instantaneous value. Higher values indicate higher utilization' +
             'of the floating-point cores.'],
        [ 'DRAM Active (%)',
             'The fraction of cycles where data was sent to or received from' +
             'device memory. The value represents an average over a time' +
             'interval and is not an instantaneous value. Higher values indicate' +
             'higher utilization of device memory. An activity of 100% is' +
             'equivalent to a DRAM instruction every cycle over the entire time' +
             'interval (in practice a peak of ~0.8 (80%) is the maximum' +
             'achievable).'],
        [ 'NVLink Send/Recv Bandwidth (byte/s)',
             'The rate of data transmitted / received over NVLink, not' +
             'including protocol headers. The theoretical maximum NVLink Gen2' +
             'bandwidth is 25 GB/s per link per direction.' ],
        [ 'PCIe Send/Recv Bandwidth (byte/s)',
             'The rate of data transmitted / received over the PCIe bus, including' +
             'both protocol headers and data payloads. The theoretical maximum PCIe' +
             'Gen3 bandwidth is 985 MB/s per lane.']
    ]

    row_line_counts = []  # to store number of lines per row
    for row in table_data:
        label, val = row

        val_str = str(label)
        col0 = textwrap.wrap(val_str, width=col0_width, max_lines=20)
        row[0] = '\n'.join(col0)

        val_str = str(val)
        col1 = textwrap.wrap(val_str, width=col1_width, max_lines=20)
        row[1] = '\n'.join(col1)
        row_line_counts.append(max(len(col0),len(col1)))

    # Create table
    table = ax.table(
        cellText=table_data,
        cellLoc='left',
        loc='top',
        bbox = (0,0,1,1),
        colWidths = [0.4, 0.7]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)

    for i, lines in enumerate(row_line_counts):  # header row = 1 line
        table[i, 0].set_height(0.01 * lines)
        table[i, 1].set_height(0.01 * lines)

    # --- Apply alternating row colors ---
    colors = ['#ffffff', '#dddddd']  # white and light grey
    for i in range(0, len(row_line_counts)): 
        color = colors[(i+1) % 2]  # alternate colors
        table[i, 0].set_facecolor(color)
        table[i, 1].set_facecolor(color)

    pdf.savefig(fig)
    pl.close(fig)

def plots(pdf, df):
    """
    Draw all the time-series plots of summarized GPU metrics.

    Parameters
    ----------
    pdf : 
        The open pdf file handle to save the title page figure into.
    df : pandas.DataFrame
        DataFrame containing job metrics.

    Returns
    -------
    None
    """

    pl.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 8,
        'legend.fontsize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
    })

    nr, nc = 5,2
    fig, axes = pl.subplots(nrows=nr, ncols=nc, figsize=(8, 11),
                            squeeze=False, sharey='row', 
                            gridspec_kw=dict(wspace=0.1, hspace=.4, width_ratios=[6,2]))

    fig.text(
        0.1, 0.95,
        'Aggregate GPU Performance Metrics',
        ha='left',
        va='top',
        fontsize=16,
        fontweight='bold'
    )

    fig.text(
        0.1, 0.925,
        'Data summarized over all GPUs.',
        ha='left',
        va='top',
        fontsize=10,
        #fontweight='bold'
    )

    plot_active_metrics(axes[1,:], df)
    plot_sm_metrics(axes[2,:], df)
    plot_memory_metrics(axes[0,:], df)
    plot_txrx_metrics(axes[3,:], df)
    plot_energy_metrics(axes[4,:], df)

    pdf.savefig(fig)
    pl.close(fig)

def tables(pdf, df):
    pass

def reduced_df(df):
    """ 
    Reduce a DataFrame over timestamp, find min/max/avg of columns ending in _avg
    """

    cols_to_aggregate = [col for col in df.columns if col.endswith('_avg')]

    agg_df = df.groupby('timestamp')[cols_to_aggregate].agg(['min', 'max', 'mean', 'sum'])
    agg_df.reset_index(inplace=True)
    #agg_df['DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION_avg', 'cumsum'] = agg_df['DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION_avg', 'sum'].cumsum()

    return agg_df

def heatmaps(pdf, df):
    """
    Draw a page of heatmaps for the Tensor and FP metrics

    Parameters
    ----------
    pdf : 
        The open pdf file handle to save the title page figure into.
    df : pandas.DataFrame
        DataFrame containing job metrics.

    Returns
    -------
    None
    """

    pl.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 8,
        'legend.fontsize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
    })

    nr, nc = 4,1
    fig, axes = pl.subplots(nrows=nr, ncols=nc, figsize=(8, 11),
                            squeeze=False, sharey='row', 
                            gridspec_kw=dict(wspace=0.1, hspace=.4, width_ratios=[1]))
    fig.text(
        0.1, 0.95,
        'Individual GPU Performance Metrics',
        ha='left',
        va='top',
        fontsize=16,
        fontweight='bold'
    )

    fig.text(
        0.1, 0.925,
        'Data from all GPUs.',
        ha='left',
        va='top',
        fontsize=10,
        #fontweight='bold'
    )

    cfg = [
        ['DCGM_FI_PROF_PIPE_TENSOR_ACTIVE_avg', 100, 'm', 'Tensor', 'Purples'], 
        ['DCGM_FI_PROF_PIPE_FP64_ACTIVE_avg',   100, 'r', 'fp64',   'Reds'],
        ['DCGM_FI_PROF_PIPE_FP32_ACTIVE_avg',   100, 'g', 'fp32',   'Greens'],
        ['DCGM_FI_PROF_PIPE_FP16_ACTIVE_avg',   100, 'b', 'fp16',   'Blues'],
        ]

    from packaging import version
    if version.parse(mpl.__version__) >= version.parse("3.7"):
        cmap_func = mpl.colormaps.get_cmap
    else:
        cmap_func = cm.get_cmap


    for i,[metric,scale,c,label,cmap] in enumerate(cfg):
        ax1 = axes[i,0]
        xy = df.pivot(
            index=['proc', 'gpuId'],   # unique GPU identifier
            columns='timestamp',       # columns = time
            values=metric              # values to populate the matrix
        )
        #x = xy.columns
        x = range(len(xy.columns)+1)
        y = range(len(xy.index)+1)
        vals = xy.fillna(0.0).to_numpy() * scale

        #x = x[700:801]
        #vals = vals[:,700:800]
    

        clrs = copy.copy(cmap_func('viridis'))
        #clrs = copy.copy(cmap_func('plasma'))
        #clrs = copy.copy(cmap_func(cmap))
        clrs.set_under('white')
        clrs.set_over('red')

        im = ax1.pcolormesh(
            x,
            y,
            vals,
            shading="flat",
            vmin=1, vmax=100, cmap=clrs,
            rasterized=True,
            snap=True,
        )

        cbar = pl.colorbar(im, ax=ax1, extend='both')

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel(f'GPU Index', labelpad=10)
        ax1.set_title(f'{label} activity (%)')
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))


    pdf.savefig(fig)
    pl.close(fig)


def evaluation(pdf, df, metadf):
    """
    Evaluate the metrics to give an analysis and recommendations about the job.

    Parameters
    ----------
    pdf : 
        The open pdf file handle to save the title page figure into.
    df : pandas.DataFrame
        DataFrame containing job metrics.
    metadf : pandas.DataFrame
        DataFrame containing job metadata.

    Returns
    -------
    None
    """

    fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(8, 11), squeeze=False)
    ax = axes[0,0]

    ax.axis('off')  # hide axes

    fig.text(
        0.1, 0.95,
        'Evaluation (Experimental)',
        ha='left',
        va='top',
        fontsize=16,
        fontweight='bold'
    )

    col0_width = 25
    col1_width = 45

    colors = dict(
            good = 'g',
            acceptable = 'b',
            improve = 'orange',
            poor = 'r'
            )

    cfg = [
        [df['DCGM_FI_DEV_GPU_UTIL_avg'].mean(), '%.2f',
            [[ 25, 'poor',        'The global average GPU utilization is poor and below acceptable use of node resources.'],
             [ 50, 'improve',     'The global average GPU utilization is below acceptable limits.'],
             [ 75, 'acceptable',  'The global average GPU utilization could be improved.'],
             [100, 'good',        'The global average GPU utilization is a good use of node resources.'],
            ]
        ],

        [df[['DCGM_FI_PROF_PIPE_TENSOR_ACTIVE_avg',
            'DCGM_FI_PROF_PIPE_FP64_ACTIVE_avg',
            'DCGM_FI_PROF_PIPE_FP32_ACTIVE_avg',
            'DCGM_FI_PROF_PIPE_FP16_ACTIVE_avg']].sum(axis=1).mean(), '%.2f',
            [[1/16., 'poor',        'The global average FP utilization is poor and below acceptable use of node resources.'],
             [1/8. , 'improve',     'The global average FP utilization is below acceptable limits.'],
             [1/4. , 'acceptable',  'The global average FP utilization could be improved.'],
             [4    , 'good',        'The global average FP utilization is a good use of node resources.'],
            ] 
        ]
    ]

    table_colors = []
    table_data = []
    for c in cfg:
        val, fmt, limits = c
        for [lim, rating, msg] in limits:
            if val <= lim:
                table_data.append(['', fmt % val, msg])
                table_colors.append(colors[rating])
                break

    row_line_counts = []  # to store number of lines per row
    for row in table_data:
        _, label, msg = row
        col0 = textwrap.wrap(str(label), width=col0_width, max_lines=20)
        col1 = textwrap.wrap(str(msg),   width=col1_width, max_lines=20)

        row[1] = '\n'.join(col0)
        row[2] = '\n'.join(col1)

        row_line_counts.append(max(len(col0),len(col1)))

    # Create table
    h = 0
    for i, lines in enumerate(row_line_counts):  # header row = 1 line
        h += 0.05 * lines

    table = ax.table(
        cellText=table_data,
        cellLoc='left',
        loc='top',
        bbox = (0,1-h,1,h),
        colWidths = [0.1, 0.4, 0.7]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for i, lines in enumerate(row_line_counts):  # header row = 1 line
        table[i, 0].set_height(0.01 * lines)
        table[i, 1].set_height(0.01 * lines)
        table[i, 2].set_height(0.01 * lines)

    # --- Apply alternating row colors ---
    colors = ['#ffffff', '#dddddd']  # white and light grey
    for i in range(0, len(row_line_counts)): 
        color = table_colors[i]  # alternate colors
        table[i, 0].set_facecolor(color)
        color = colors[(i+1) % 2]  # alternate colors
        table[i, 1].set_facecolor(color)
        table[i, 2].set_facecolor(color)

    pdf.savefig(fig)
    pl.close(fig)


def one_report(pdf, metadf, df):
    """
    Create a single report for the given metric data

    Parameters
    ----------
    pdf : 
        The pdf object to write to
    df : pandas.DataFrame
        DataFrame containing job metrics.
    metadf : pandas.DataFrame
        DataFrame containing job metadata.

    Returns
    -------
    None
    """

    if df is None:
        title_page(pdf, None, metadf)
        return

    rdf = reduced_df(df)

    title_page(pdf, df, metadf)
    evaluation(pdf, df, metadf)
    plots(pdf, rdf)
    heatmaps(pdf, df)
    tables(pdf, rdf)

def report(paths, pdfname):
    """
    Create a report for all job paths in paths.

    Parameters
    ----------
    paths : list
        List of paths to metric data
    pdfname : 
        The filename to use for the PDF report.

    Returns
    -------
    None
    """

    df, metadf = load_metrics_and_meta(paths)

    if metadf.empty:
        print(f'Could not load any data from given paths. No report generated.')
        return

    with PdfPages(pdfname) as pdf:

        for [group, gmdf] in metadf.groupby(['report', 'step']):
            report,step = group
            if df.empty:
                gdf = None
            else:
                gdf = df[(df["report"] == report) & (df["step"] == step)]
                gdf.reset_index(inplace=True, drop=True)

            one_report(pdf, gmdf, gdf)

        definitions_page(pdf)

    print(f'Saved PDF report to {pdfname}')

def verify_input_directories(paths):
    """
    Sanity check that the directories given by the user exist

    Parameters
    ----------
    paths : list
        List of paths to metric data

    Returns
    -------
    Boolean
    """

    ok = True

    for path in paths:
        if not os.path.exists(path):
            print(f'The directory {path} does not exist.')
            ok = False
        elif not os.path.isdir(path):
            print(f'The given location {path} is not a directory.')
            ok = False

    return ok

def main(args):
    """
    Main driver to parse arguments, load data, and make the report

    Parameters
    ----------
    args : 
        Command line arguments.

    Returns
    -------
    None
    """
    args = parse_args(args)

    if not verify_input_directories(args.paths):
        exit(1)

    report(args.paths, args.output)


def parse_args(args):
    """
    Parse commandline arguments

    Parameters
    ----------
    args : 
        Command line arguments.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description='Generate a report from GPU metrics collected with gssr-record.'
    )

    # Required argument for root directory
    parser.add_argument(
        'paths',
        metavar='directory',
        nargs='*',
        help='Top level directories created by gssr-record'
    )

    # Optional argument for output PDF
    parser.add_argument(
        '-o', '--output',
        default='gssr-report.pdf',
        help='Output PDF filename (default: gssr-report.pdf)'
    )

    args = parser.parse_args(args)

    if len(args.paths) == 0:
        parser.print_help()
        sys.exit(1)

    return args


if __name__ == '__main__':
    main(sys.argv[1:])

