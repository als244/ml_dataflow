import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np


device_name_to_metric_ranges = {
    'RTX 3090': {
        'tok_per_sec': {'vmin': 3000, 'vmax': 23000},
        'tflops':      {'vmin': 80, 'vmax': 209.5},
        'mfu':         {'vmin': .3, 'vmax': 1.0}
    },
    'RTX 5090': {
        'tok_per_sec': {'vmin': 3000, 'vmax': 23000},
        'tflops':      {'vmin': 80, 'vmax': 209.5},
        'mfu':         {'vmin': .3, 'vmax': 1.0}
    },
    'A100': {
        'tok_per_sec': {'vmin': 3000, 'vmax': 23000},
        'tflops':      {'vmin': 80, 'vmax': 209.5},
        'mfu':         {'vmin': .3, 'vmax': 1.0}
    },
    'H100': {
        'tok_per_sec': {'vmin': 3000, 'vmax': 23000},
        'tflops':      {'vmin': 80, 'vmax': 209.5},
        'mfu':         {'vmin': .3, 'vmax': 1.0}
    }
}

# ---

def plot_throughput(csv_filepath, device_name, output_dir):
    """
    Generates heatmaps with a custom color map and user-defined color ranges.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Style Settings ---
    # Global font sizes
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    annot_kws = {"size": 14}

    # --- New: Create a custom colormap from yellow to green ---
    custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "salmon_to_lawngreen", ["salmon", "lawngreen"]
    )
    # ---

    df = pd.read_csv(csv_filepath, names=['host_mem_gb', 'dev_mem_gb', 'seq_len', 'model_size', 'seqs_per_step', 'avg_step_time', 'tok_per_sec', 'tflops', 'mfu'])

    metrics = ['tok_per_sec', 'tflops', 'mfu']
    metric_labels = {
        'tok_per_sec': 'Tokens per Second',
        'tflops': 'TFLOPS',
        'mfu': 'MFU'
    }

    seq_lens = df['seq_len'].unique()
    model_sizes = df['model_size'].unique()

    METRIC_RANGES = device_name_to_metric_ranges[device_name]

    for seq_len in seq_lens:
        df_seqlen_slice = df[df['seq_len'] == seq_len]
        for model_size in model_sizes:
            df_model_size_slice = df_seqlen_slice[df_seqlen_slice['model_size'] == model_size]

            if df_model_size_slice.empty:
                continue

            for metric in metrics:
                global_vmin = METRIC_RANGES[metric]['vmin']
                global_vmax = METRIC_RANGES[metric]['vmax']

                heatmap_data = df_model_size_slice.pivot_table(
                    index='host_mem_gb',
                    columns='dev_mem_gb',
                    values=metric,
                    aggfunc='mean'
                ).fillna(0)
                
                # --- CHANGE: Sort the index to control Y-axis order ---
                # Sorting descending places the largest value first, which goes on top.
                heatmap_data.sort_index(ascending=False, inplace=True)
                # ---

                if heatmap_data.empty:
                    continue

                plt.figure(figsize=(10, 8))

                hide_zeros = heatmap_data == 0
                hide_non_zeros = heatmap_data != 0
                dark_red_cmap = matplotlib.colors.ListedColormap(['#8B0000'])

                ax = sns.heatmap(
                    heatmap_data,
                    mask=hide_zeros,
                    annot=True,
                    annot_kws=annot_kws,
                    fmt=".2f",
                    linewidths=1.0,
                    linecolor='white',
                    cmap=custom_cmap,
                    vmin=global_vmin,
                    vmax=global_vmax,
                    cbar_kws={'label': metric_labels[metric]}
                )
                
                # The ax.invert_yaxis() call has been removed.

                sns.heatmap(
                    heatmap_data,
                    mask=hide_non_zeros,
                    annot=True,
                    annot_kws=annot_kws,
                    fmt=".2f",
                    linewidths=1.0,
                    linecolor='white',
                    cmap=dark_red_cmap,
                    cbar=False,
                    ax=ax
                )

                plt.title(f"Performance for Model: {model_size}, Seq Len: {seq_len}")
                plt.xlabel("Device Memory (GB)")
                plt.ylabel("Host Memory (GB)")

                plt.tight_layout()

                output_filename = f"model_{model_size}_seqlen_{seq_len}_{metric}.pdf"
                output_path = os.path.join(output_dir, output_filename)

                plt.savefig(output_path)
                plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python plot_throughput.py <csv filepath to plot> <device name> <output dir>")
        sys.exit(1)

    path_to_results = sys.argv[1]
    device_name = sys.argv[2]
    output_dir = sys.argv[3]
    print(f"Plotting throughput results from {path_to_results} and saving plots to {output_dir}")
    plot_throughput(path_to_results, device_name, output_dir)
    print("Done.")