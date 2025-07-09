import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

def plot_throughput(csv_filepath, device_name, output_dir):
    """
    Generates heatmaps with metric-specific colormaps and value ranges.
    - Tokens/sec: YlGn, vmin=min_non_zero, vmax=max
    - TFLOPS: Blues, vmin=0, vmax=peak_tflops
    - MFU/HFU: RdYlGn, vmin=0, vmax=1
    - Zero values are colored dark red for emphasis across all plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    ## THESE AREN'T REAL PEAK FLOPS,
    ## BUT JUST A PROXY FOR MAX ATTAINABLE FLOPS
    device_name_to_peak_tflops = {
        "H100": 600,
        "A100": 200,
        "RTX5090": 190,
        "RTX3090": 60
    }

    if device_name not in device_name_to_peak_tflops:
        raise ValueError(f"Device name {device_name} not found in device_name_to_peak_tflops")

    peak_tflops = device_name_to_peak_tflops[device_name]

    # --- Style Settings ---
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14
    })
    annot_kws = {"size": 14}

    # --- Colormap for zero values ---
    dark_red_cmap = matplotlib.colors.ListedColormap(['#8B0000'])

    csv_columns = ["host_mem_gb", "dev_mem_gb", "seq_len", "model_size", "chunk_size", "total_home_acts", "num_inp_only_saved", "num_inp_attn_saved", "num_full_saved", "total_dev_acts", "num_rounds_per_step", "seqs_per_step", "recompute_pct", "attn_flop_pct", "avg_step_time", "tok_per_sec", "tflops", "mfu", "hfu"]
    df = pd.read_csv(csv_filepath, names=csv_columns)

    metrics = ['tok_per_sec', 'tflops', 'mfu', 'hfu']
    metric_labels = {
        'tok_per_sec': 'Tokens per Second',
        'tflops': 'TFLOPS',
        'mfu': 'MFU',
        'hfu': 'HFU',
    }
    metric_file_suffix = {
        'tok_per_sec': 'tok',
        'tflops': 'tflops',
        'mfu': 'mfu',
        'hfu': 'hfu',
    }

    seq_lens = df['seq_len'].unique()
    model_sizes = df['model_size'].unique()

    for seq_len in seq_lens:
        df_seqlen_slice = df[df['seq_len'] == seq_len]
        for model_size in model_sizes:
            df_model_size_slice = df_seqlen_slice[df_seqlen_slice['model_size'] == model_size]

            if df_model_size_slice.empty:
                continue

            for metric in metrics:
                if metric not in df_model_size_slice.columns:
                    continue

                heatmap_data = df_model_size_slice.pivot_table(
                    index='host_mem_gb',
                    columns='dev_mem_gb',
                    values=metric,
                    aggfunc='mean'
                ).fillna(0)

                heatmap_data.sort_index(ascending=False, inplace=True)

                if heatmap_data.empty:
                    continue

                # --- Dynamic Colormap and Range Settings ---
                if metric == 'tflops':
                    cmap = 'Blues'
                    vmin = 0
                    vmax = peak_tflops
                elif metric in ['mfu', 'hfu']:
                    cmap = 'RdYlGn'
                    vmin = 0
                    vmax = 1
                elif metric == 'tok_per_sec':
                    cmap = 'YlGn'
                    non_zero_vals = heatmap_data[heatmap_data > 0]
                    # Set vmin to the smallest non-zero value, or 0 if all are zero
                    vmin = non_zero_vals.min().min() if not non_zero_vals.empty else 0
                    vmax = heatmap_data.max().max()
                # --- End of Dynamic Settings ---

                plt.figure(figsize=(10, 8))

                # Create masks for zero and non-zero values to plot them separately
                hide_zeros = heatmap_data == 0
                hide_non_zeros = heatmap_data != 0

                # Plot the main heatmap for non-zero values
                ax = sns.heatmap(
                    heatmap_data,
                    mask=hide_zeros,
                    annot=True,
                    annot_kws=annot_kws,
                    fmt=".2f",
                    linewidths=1.0,
                    linecolor='white',
                    cmap=cmap,
                    cbar_kws={'label': metric_labels.get(metric, metric)},
                    vmin=vmin,
                    vmax=vmax
                )

                # Overlay a heatmap for zero values using the dark red color
                sns.heatmap(
                    heatmap_data,
                    mask=hide_non_zeros,
                    annot=False,
                    linewidths=1.0,
                    linecolor='white',
                    cmap=dark_red_cmap,
                    cbar=False, # No colorbar needed for the zero values
                    ax=ax
                )

                plt.title(f"Performance for Model: {model_size}B, Seq Len: {seq_len} on {device_name}")
                plt.xlabel("Device Memory (GB)")
                plt.ylabel("Host Memory (GB)")
                plt.tight_layout()

                output_filename = f"{device_name}-{model_size}B-{seq_len}-{metric_file_suffix[metric]}.pdf"
                output_path = os.path.join(output_dir, output_filename)

                plt.savefig(output_path)
                plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python plot_throughput.py <csv filepath to plot> <device_name> <output dir>")
        sys.exit(1)

    path_to_results = sys.argv[1]
    device_name = sys.argv[2]
    output_dir = sys.argv[3]
    print(f"Plotting throughput results from {path_to_results} for {device_name} and saving plots to {output_dir}")
    plot_throughput(path_to_results, device_name, output_dir)
    print("Done.")