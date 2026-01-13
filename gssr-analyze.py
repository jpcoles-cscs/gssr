#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.6"
# dependencies = [
#   "numpy",
#   "matplotlib",
#   "pandas",
# ]
# ///

import sys,os
import argparse
import numpy as np
import pandas as pd
import json
import textwrap
import copy
from matplotlib import cm
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages

rng = np.random.default_rng()


def load_csv_tree(root, label_column='source'):
    """
    Walk a directory tree and load all CSV files into a single DataFrame.

    Parameters
    ----------
    root : str
        Root directory to walk.
    label_column : str
        Name of the column used to label rows by filename.

    Returns
    -------
    pandas.DataFrame
    """
    frames = []
    meta = []

    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)

            if name.lower().endswith('meta.txt'):

                print(f'Found metatdata {name}')
                with open(path) as f:
                    df = pd.DataFrame([json.load(f)])
                    label = f'{dirpath}_{name}'
                    df[label_column] = label
                    meta.append(df)
                continue

            if not name.lower().endswith('.csv'):
                continue


            try:
                print(f'Found {name}')
                df = pd.read_csv(path)

                # Use filename (without extension) as label
                label = f'{dirpath}_' + os.path.splitext(name)[0]
                df[label_column] = label

                frames.append(df)
            except pd.errors.EmptyDataError:
                pass


    if not frames:
        df = pd.DataFrame()
    else:
        df = pd.concat(frames, ignore_index=True)

    if not meta:
        metadf = pd.DataFrame()
    else:
        metadf = pd.concat(meta, ignore_index=True)

    df[['report']] = df['source'].str.extract(r'^(.+)/').astype(str)
    df[['step', 'proc']] = df['source'].str.extract(
        r'^.+/step_(\d+)_proc_(\d+)'
    ).astype(int)

    return df, metadf

def plot_memory_metrics(ax, df):

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

    cfg = [
        ['DCGM_FI_PROF_PCIE_TX_BYTES_avg',      1e-6, 'g',        'PCIe Send'], 
        ['DCGM_FI_PROF_PCIE_RX_BYTES_avg',      1e-6, 'r',        'PCIe Recv'],
        ['DCGM_FI_PROF_NVLINK_TX_BYTES_avg',    1e-6, 'b',        'NVLink Send'],
        ['DCGM_FI_PROF_NVLINK_RX_BYTES_avg',    1e-6, 'orange',   'NVLink Recv']]

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
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel(f'Data Movement (MB/s)', labelpad=10)

        n_bins = 20 
        y,bins = np.histogram(y_avg, weights=np.full_like(y_avg, 1./len(y_avg)) * 100, bins=n_bins, range=(0,100))
        yeps = rng.integers(low=0, high=2, size=len(y))
        yeps[y <= 0] = 0
        xeps = rng.integers(low=0, high=2, size=len(y))
        ax[1].plot(y + yeps, bins[0:-1] + xeps, alpha=0.8, color=c, lw=1.0, drawstyle='steps')

    ax[1].set_title('Histogram', fontsize=6)
    ax[1].set_xlabel('% of Runtime')
    ax[1].set_xticks([0,25,50,75,100])
    ax[1].set_xlim(-5, 105)
    ax[1].set_ylim(ax[0].get_ylim())
    
    legend = [ [legend_text,   pl.Line2D([0], [0], lw=5, color=c)] for _,_,c,legend_text in cfg ]
    l,h = list(zip(*legend))
    ax[0].legend(h,l, ncol=len(legend), loc='lower left', borderpad=0, **dict(frameon=False, bbox_to_anchor=(0.0, 1.0), fontsize=6))

def plot_active_metrics(ax, df):

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
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel(f'GPU activity (%)', labelpad=10)

        # Create the distribution plot (right)
        n_bins = 20 #min(100, max(10, int(np.ceil(np.sqrt(len(y_avg))))))
        y,bins = np.histogram(y_avg, weights=np.full_like(y_avg, 1./len(y_avg)) * 100, bins=n_bins, range=(0,100))

        yeps = rng.integers(low=0, high=2, size=len(y))
        yeps[y <= 0] = 0
        xeps = rng.integers(low=0, high=2, size=len(y))
        ax[1].plot(y + yeps, bins[0:-1] + xeps, alpha=0.8, color=c, lw=1.0, drawstyle='steps')

    ax[0].set_ylim(ymin=0, ymax=100)

    ax[1].set_title('Histogram', fontsize=6)
    ax[1].set_xlabel('% of Runtime')
    ax[1].set_xticks([0,25,50,75,100])
    ax[1].set_xlim(-5, 105)
    ax[1].set_ylim(ax[0].get_ylim())
    

    legend = [ [legend_text,   pl.Line2D([0], [0], lw=5, color=c)] for _,_,c,legend_text in cfg ]
    l,h = list(zip(*legend))
    ax[0].legend(h,l, ncol=len(legend), loc='lower left', borderpad=0, **dict(frameon=False, bbox_to_anchor=(0.0, 1.0), fontsize=6))

def plot_sm_metrics(ax, df):

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
    ax[0].set_ylim(ymin=0, ymax=100)
    
    ax[1].set_title('Histogram', fontsize=6)
    ax[1].set_xlabel('% of Runtime')
    ax[1].set_xticks([0,25,50,75,100])
    ax[1].set_xlim(-5, 105)
    ax[1].set_ylim(ax[0].get_ylim())

    
    legend = [ [legend_text,   pl.Line2D([0], [0], lw=5, color=c)] for _,_,c,legend_text in cfg ]
    l,h = list(zip(*legend))
    ax[0].legend(h,l, ncol=len(legend), loc='lower left', borderpad=0, **dict(frameon=False, bbox_to_anchor=(0.0, 1.0), fontsize=6))

def title_page_table(ax, metadf):

    wrap_width = 45  # wrap lines at 30 characters

    m = metadf.iloc[0].to_dict()

    table_data = [
        [ 'Slurm Job ID',   m.get('jobid',      'missing')  ],
        [ 'Cluster',        m.get('cluster',    'missing')  ],
        [ 'Date',           m.get('date',       'missing')  ],
        [ 'Job Name',       m.get('jobname',    'missing')  ],
        [ 'Node Count',     m.get('nnodes',     'missing')  ],
        [ 'Task Count',     m.get('ntasks',     'missing')  ],
        [ 'GPU Count',      m.get('ngpus',      'missing')  ],
        [ 'Executable',     m.get('executable', 'missing')  ],
        [ 'Arguments',      m.get('arguments',  'missing')  ],
    ]

    row_line_counts = []  # to store number of lines per row
    for row in table_data:
        label, val = row
        val_str = str(val)
        wrapped_val = textwrap.wrap(val_str, width=wrap_width, max_lines=20)
        row[1] = '\n'.join(wrapped_val)
        row_line_counts.append(len(wrapped_val))

    # Create table
    table = ax.table(
        cellText=table_data,
        #colLabels=['Field', 'Value'],
        cellLoc='left',
        loc='top',
        bbox = (0,0,1,0.5),
        colWidths = [0.3, 0.7]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for i, lines in enumerate(row_line_counts):  # header row = 1 line
        table[i, 0].set_height(0.04 * lines)
        table[i, 1].set_height(0.04 * lines)

    # --- Apply alternating row colors ---
    colors = ['#ffffff', '#dddddd']  # white and light grey
    for i in range(0, len(row_line_counts)):  # skip header row
        color = colors[(i-1) % 2]  # alternate colors
        table[i, 0].set_facecolor(color)
        table[i, 1].set_facecolor(color)


def title_page(pdf, df, metadf):

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

    title_page_table(ax, metadf)

    pdf.savefig(fig)
    pl.close(fig)

    pass

def definitions_page(pdf, df):

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


    col0_width = 25
    col1_width = 45

    table_data = [
        [ 'GPU Utilization (%)',                  'Percent of time over the processÕs lifetime during which one or more kernels was executing on the GPU.' ],
        [ 'SM Active (%)',                        'The ratio of cycles an SM has at least 1 warp assigned (computed from the number of cycles and elapsed cycles)' ],
        [ 'SM Occupancy (%)',                     'Occupancy is defined as the ratio of active warps on an SM to the maximum number of active warps supported by the SM' ],
        [ 'Tensor Active (%)',                    'The ratio of cycles the any tensor pipe is active (off the peak sustained elapsed cycles)' ],
        [ 'FP Active (%) for FP64, FP32, FP16',   'Ratio of cycles of that a particular floating point (fp64, fp32 or fp16) pipe is active.' ],
        [ 'GPU Memory (MB) Used/Free/Reserved',     'Used/Free/Reserved Frame Buffer of the GPU in MB' ],
        [ 'NVLink Send/Recv (byte/s)',          'The rate of data transmitted / received over NVLink, not including protocol headers. (NVLink bandwidth)' ],
        [ 'PCIe Send/Recv bytes (byte/s)',            'The rate of data transmitted / received over the PCIe bus, including both protocol headers and data payloads. (PCIe bandwidth)' ]
    ]

    row_line_counts = []  # to store number of lines per row
    for row in table_data:
        label, val = row
        val_str = str(val)
        col1 = textwrap.wrap(val_str, width=col1_width, max_lines=20)
        row[1] = '\n'.join(col1)

        val_str = str(label)
        col0 = textwrap.wrap(val_str, width=col0_width, max_lines=20)
        row[0] = '\n'.join(col0)
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
    table.set_fontsize(10)

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
    pass

def plots(pdf, df):

    pl.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 8,
        'legend.fontsize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
    })

    nr, nc = 4,2
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

    pdf.savefig(fig)
    pl.close(fig)

def tables(pdf, df):
    pass

def reduced_df(df):
    """ 
    Reduce a DataFrame over timestamp, find min/max/avg of columns ending in _avg
    """

    cols_to_aggregate = [col for col in df.columns if col.endswith('_avg')]

    agg_df = df.groupby('timestamp')[cols_to_aggregate].agg(['min', 'max', 'mean'])
    agg_df.reset_index(inplace=True)

    return agg_df

def plot_active_metrics_hm(axes, df):

    cfg = [
        ['DCGM_FI_PROF_PIPE_TENSOR_ACTIVE_avg', 100, 'm', 'Tensor', 'Purples'], 
        ['DCGM_FI_PROF_PIPE_FP64_ACTIVE_avg',   100, 'r', 'fp64',   'Reds'],
        ['DCGM_FI_PROF_PIPE_FP32_ACTIVE_avg',   100, 'g', 'fp32',   'Greens'],
        ['DCGM_FI_PROF_PIPE_FP16_ACTIVE_avg',   100, 'b', 'fp16',   'Blues'],
        ]


    for i,[metric,scale,c,label,cmap] in enumerate(cfg):
        ax1 = axes[i,0]
        xy = df.pivot(
            index=['proc', 'gpuId'],   # unique GPU identifier
            columns='timestamp',       # columns = time
            values=metric              # values to populate the matrix
        )
        xy = xy.fillna(0.0).to_numpy() * scale
    
        clrs = copy.copy(cm.get_cmap(cmap))
        clrs.set_under('white')

        # Plot the actual y line over the shaded area
        im = ax1.imshow(xy, aspect='auto', interpolation='nearest', origin='upper', extent=(-0.5, xy.shape[1] - 0.5, xy.shape[0] - 0.5, -0.5),
                   vmin=1, vmax=100, cmap=clrs)
        cbar = pl.colorbar(im, ax=ax1, extend='both')

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel(f'GPU Index', labelpad=10)
        ax1.set_title(f'{label} activity (%)')

def heatmaps(pdf, df):

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

    #plot_memory_metrics(axes[0,:], df)
    plot_active_metrics_hm(axes, df)
    #plot_sm_metrics(axes[2,:], df)
    #plot_txrx_metrics(axes[3,:], df)

    pdf.savefig(fig)
    pl.close(fig)

def report(metadf, df, fname):
    with PdfPages(fname) as pdf:

        stepdf = df

        rdf = reduced_df(stepdf)

        title_page(pdf, rdf, metadf)
        plots(pdf, rdf)
        heatmaps(pdf, df)
        tables(pdf, rdf)
        definitions_page(pdf, rdf)

    print(f'Saved PDF report to {fname}')

def main(args):
    args = parse_args(args)

    if not os.path.exists(args.input):
        print(f'The directory {args.input} does not exist.')
        sys.exit(1)

    if not os.path.isdir(args.input):
        print(f'The given location {args.input} is not a directory.')
        sys.exit(1)


    df, metadf = load_csv_tree(args.input)

    report(metadf, df, args.output)


def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Generate a report from GPU metrics collected with gssr-record.'
    )

    # Required argument for root directory
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Root directory to search for CSV files'
    )

    # Optional argument for output PDF
    parser.add_argument(
        '-o', '--output',
        default='gssr-report.pdf',
        help='Output PDF filename (default: gssr-report.pdf)'
    )

    args = parser.parse_args(args)
    return args


if __name__ == '__main__':
    main(sys.argv[1:])

