import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

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

    # Create the 'pdf' subdirectory for PDF files
    pdf_output_dir = os.path.join(output_dir, "pdf")
    os.makedirs(pdf_output_dir, exist_ok=True)

    # --- Style Settings ---
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16
    })
    annot_kws = {"size": 14}

    # --- Colormap for zero values ---
    dark_red_cmap = matplotlib.colors.ListedColormap(['#8B0000'])

    csv_columns = ["host_mem_gb", "dev_mem_gb", "seq_len", "model_size", "chunk_size", "total_home_acts", "num_inp_only_saved", "num_inp_attn_saved", "num_full_saved", "total_dev_acts", "num_rounds_per_step", "seqs_per_step", "recompute_pct", "attn_flop_pct", "avg_step_time", "tok_per_sec", "tflops", "mfu", "hfu"]
    df = pd.read_csv(csv_filepath, names=csv_columns)

    device_name_to_util_range = {
        "H100": (0.3, 0.7),
        "A100": (0.15, 0.6),
        "RTX5090": (0.35, 0.9),
        "RTX3090": (0.35, 0.9)
    }
    
    util_min_val = device_name_to_util_range[device_name][0]
    util_max_val = device_name_to_util_range[device_name][1]

    device_name_to_peak_bf16_tflops = {
        "H100": 989,
        "A100": 312.5,
        "RTX5090": 209.5,
        "RTX3090": 71.2
    }

    if device_name not in device_name_to_peak_bf16_tflops:
        raise ValueError(f"Device {device_name} not found in device_name_to_peak_bf16_tflops")

    peak_bf16_tflops = device_name_to_peak_bf16_tflops[device_name]

    
    tflops_vmin = util_min_val * peak_bf16_tflops
    tflops_vmax = util_max_val * peak_bf16_tflops


    metrics = ['tok_per_sec', 'tflops', 'mfu', 'hfu']
    metric_labels = {
        'tok_per_sec': 'Tokens/sec',
        'tflops': 'TFLOPS/s',
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

            max_mfu = df_model_size_slice[df_model_size_slice['mfu'] > 0]['mfu'].max()
            max_tok_per_sec = df_model_size_slice[df_model_size_slice['tok_per_sec'] > 0]['tok_per_sec'].max()

            tok_per_sec_at_max_mfu = (util_max_val / max_mfu) * max_tok_per_sec
            tok_per_sec_at_min_mfu = (util_min_val / util_max_val) * tok_per_sec_at_max_mfu
            
            vmin_tok_per_sec = tok_per_sec_at_min_mfu
            vmax_tok_per_sec = tok_per_sec_at_max_mfu

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

                if metric == 'tflops':
                    cmap = 'viridis'
                    vmin = tflops_vmin
                    vmax = tflops_vmax
                elif metric in ['mfu', 'hfu']:
                    cmap = 'RdYlGn'
                    vmin = util_min_val
                    vmax = util_max_val
                elif metric == 'tok_per_sec':
                    cmap = 'YlGn'
                    non_zero_vals = heatmap_data[heatmap_data > 0]
                    vmin = vmin_tok_per_sec
                    vmax = vmax_tok_per_sec

                plt.figure(figsize=(10, 8))

                hide_zeros = heatmap_data == 0
                hide_non_zeros = heatmap_data != 0
                
                # --- FIX START ---
                # Prepare a dataframe for annotations to ensure all non-zero cells are annotated.
                # This explicitly tells seaborn what to print, bypassing an issue where
                # annot=True fails to add text for certain data values.
                annot_data = heatmap_data.applymap(lambda v: f"{v:.2f}" if v != 0 else "")

                ax = sns.heatmap(
                    heatmap_data,
                    mask=hide_zeros,
                    annot=annot_data,    # Use the prepared dataframe for annotations
                    fmt='',              # Disable automatic formatting, as we did it manually
                    annot_kws=annot_kws,
                    linewidths=1.0,
                    linecolor='white',
                    cmap=cmap,
                    cbar_kws={'label': metric_labels.get(metric, metric)},
                    vmin=vmin,
                    vmax=vmax
                )
                # --- FIX END ---

                # --- START: Manual Font Color Correction ---
                # This block will now work correctly because ax.texts
                # will contain an entry for every non-zero cell.
                luminance_threshold = 0.5
                
                cmap_obj = plt.get_cmap(cmap)
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
                
                annotated_data = heatmap_data.to_numpy()[hide_non_zeros.to_numpy()]
                
                for text_artist, value in zip(ax.texts, annotated_data):
                    bg_color = cmap_obj(norm(value))
                    luminance = (bg_color[0] * 299 + bg_color[1] * 587 + bg_color[2] * 114) / 1000
                    text_artist.set_color('black' if luminance > luminance_threshold else 'white')
                # --- END: Manual Font Color Correction ---

                sns.heatmap(
                    heatmap_data,
                    mask=hide_non_zeros,
                    annot=False,
                    linewidths=1.0,
                    linecolor='white',
                    cmap=dark_red_cmap,
                    cbar=False,
                    ax=ax
                )

                plt.title(f"Performance for Model: {model_size}B, Seq Len: {seq_len} on {device_name}")
                plt.xlabel("Device Memory (GB)")
                plt.ylabel("Host Memory (GB)")
                plt.tight_layout()

                base_filename = f"{device_name}-{model_size}B-{seq_len}-{metric_file_suffix[metric]}"
                pdf_path = os.path.join(pdf_output_dir, f"{base_filename}.pdf")
                plt.savefig(pdf_path)
                png_path = os.path.join(output_dir, f"{base_filename}.png")
                plt.savefig(png_path)
                
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