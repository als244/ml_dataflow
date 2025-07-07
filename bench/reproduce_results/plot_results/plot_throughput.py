import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

def plot_throughput(csv_filepath, device_name, output_dir):
    """
    Generates heatmaps with a custom pink-to-green color map and automatically inferred color ranges.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Style Settings ---
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14
    })
    annot_kws = {"size": 14}

    # --- Create a custom colormap from pink to green ---
    custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "pink_to_green", ["pink", "green"]
    )

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
                # Skip metrics that might not be in the dataframe
                if metric not in df_model_size_slice.columns:
                    continue

                heatmap_data = df_model_size_slice.pivot_table(
                    index='host_mem_gb',
                    columns='dev_mem_gb',
                    values=metric,
                    aggfunc='mean'
                ).fillna(0)

                # Sort the index to have the largest value on top
                heatmap_data.sort_index(ascending=False, inplace=True)

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
                    cmap=custom_cmap, # Use the new colormap
                    cbar_kws={'label': metric_labels.get(metric, metric)}
                    # vmin and vmax are removed to allow automatic scaling
                )

                sns.heatmap(
                    heatmap_data,
                    mask=hide_non_zeros,
                    annot=False, # No need to annotate twice
                    linewidths=1.0,
                    linecolor='white',
                    cmap=dark_red_cmap,
                    cbar=False,
                    ax=ax
                )

                plt.title(f"Performance for Model: {model_size}, Seq Len: {seq_len} on {device_name}")
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