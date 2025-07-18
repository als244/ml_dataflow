import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

def plot_throughput(csv_filepath, device_name, output_dir):
    """
    Generates a single 2x2 subplot image and PDF containing four heatmaps 
    (Tokens/sec, TFLOPS, MFU, HFU) for each model size and sequence length.
    - The figure is titled with device, model, and sequence length information.
    - Each subplot is titled with its corresponding metric.
    - Metric-specific colormaps and value ranges are applied.
    - Zero values are colored black for emphasis across all plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_output_dir = os.path.join(output_dir, "pdf")
    os.makedirs(pdf_output_dir, exist_ok=True)

    # --- Style and Colormap Settings ---
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'figure.titlesize': 24
    })
    annot_kws = {"size": 16}
    failed_cmap = matplotlib.colors.ListedColormap(['#000000'])

    # --- Data Loading and Preparation ---
    csv_columns = ["host_mem_gb", "dev_mem_gb", "seq_len", "model_size", "used_host_mem_gb", "used_dev_mem_gb", "chunk_size", "total_home_acts", "num_inp_only_saved", "num_inp_attn_saved", "num_full_saved", "total_dev_acts", "num_rounds_per_step", "seqs_per_step", "recompute_pct", "attn_flop_pct", "avg_step_time", "tok_per_sec", "tflops", "mfu", "hfu"]
    
    df = pd.read_csv(csv_filepath, names=csv_columns)

    # --- Device-Specific Performance Ranges ---
    device_name_to_util_range = {
        "H100": (0.1, 0.9), "A100": (0.1, 0.9),
        "RTX5090": (0.1, 0.9), "RTX3090": (0.1, 0.9)
    }
    util_min_val, util_max_val = device_name_to_util_range.get(device_name, (0.1, 0.9))

    device_name_to_peak_bf16_tflops = {
        "H100": 989, "A100": 312.5,
        "RTX5090": 209.5, "RTX3090": 71.2
    }
    if device_name not in device_name_to_peak_bf16_tflops:
        raise ValueError(f"Device {device_name} not found in device_name_to_peak_bf16_tflops")
    peak_bf16_tflops = device_name_to_peak_bf16_tflops[device_name]
    
    tflops_vmin = util_min_val * peak_bf16_tflops
    tflops_vmax = util_max_val * peak_bf16_tflops

    # --- Metric Definitions ---
    metric_labels = {
        'tok_per_sec': 'Tokens/sec', 'tflops': 'Model TFLOPS/sec',
        'mfu': 'MFU', 'hfu': 'HFU',
    }
    plot_positions = {
        (0, 0): 'tok_per_sec', (1, 0): 'tflops',
        (0, 1): 'mfu', (1, 1): 'hfu',
    }

    # --- Main Plotting Loop ---
    seq_lens = df['seq_len'].unique()
    model_sizes = df['model_size'].unique()

    for seq_len in seq_lens:
        df_seqlen_slice = df[df['seq_len'] == seq_len]
        for model_size in model_sizes:
            df_model_size_slice = df_seqlen_slice[df_seqlen_slice['model_size'] == model_size]            

            if df_model_size_slice.empty:
                continue

            # --- Create a 2x2 Subplot Figure ---
            fig, axs = plt.subplots(2, 2, figsize=(22, 18), sharex=True, sharey=True)
            fig.suptitle(f"{device_name} Performance Report\nModel: {model_size}B, Sequence Length: {seq_len}")

            # Calculate tok/sec range based on MFU
            max_mfu = df_model_size_slice[df_model_size_slice['mfu'] > 0]['mfu'].max()
            max_tok_per_sec = df_model_size_slice[df_model_size_slice['tok_per_sec'] > 0]['tok_per_sec'].max()
            tok_per_sec_at_max_mfu = (util_max_val / max_mfu) * max_tok_per_sec if max_mfu > 0 else 0
            tok_per_sec_at_min_mfu = (util_min_val / util_max_val) * tok_per_sec_at_max_mfu
            
            # --- Loop Through Subplot Positions and Metrics ---
            for (r, c), metric in plot_positions.items():
                ax = axs[r, c]
                if metric not in df_model_size_slice.columns:
                    ax.text(0.5, 0.5, f"{metric_labels[metric]}\nData Not Available", ha='center', va='center', fontsize=18)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                heatmap_data = df_model_size_slice.pivot_table(
                    index='host_mem_gb', columns='dev_mem_gb', values=metric, aggfunc='mean'
                ).fillna(0)
                heatmap_data.sort_index(ascending=False, inplace=True)

                if heatmap_data.empty:
                    continue

                # --- Define Colormap and Value Range for Each Metric ---
                if metric == 'tflops':
                    cmap, vmin, vmax = 'viridis', tflops_vmin, tflops_vmax
                elif metric in ['mfu', 'hfu']:
                    cmap, vmin, vmax = 'RdYlGn', util_min_val, util_max_val
                elif metric == 'tok_per_sec':
                    cmap, vmin, vmax = 'YlGn', tok_per_sec_at_min_mfu, tok_per_sec_at_max_mfu

                # Prepare masks and annotation data
                mask_zeros = heatmap_data == 0
                annot_data = heatmap_data.applymap(lambda v: f"{v:.2f}" if v != 0 else "")

                # Plot heatmap for non-zero values
                sns.heatmap(
                    heatmap_data, mask=mask_zeros, annot=annot_data, fmt='', annot_kws=annot_kws,
                    linewidths=1.0, linecolor='white', cmap=cmap, vmin=vmin, vmax=vmax,
                    cbar_kws={'label': metric_labels.get(metric, metric)}, ax=ax
                )
                
                # --- Manual Font Color Correction for Readability ---
                cmap_obj = plt.get_cmap(cmap)
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
                for text_artist in ax.texts:
                    val = float(text_artist.get_text())
                    bg_color = cmap_obj(norm(val))
                    luminance = (bg_color[0] * 299 + bg_color[1] * 587 + bg_color[2] * 114) / 1000
                    text_artist.set_color('black' if luminance > 0.5 else 'white')

                # Overlay heatmap for zero values
                sns.heatmap(
                    heatmap_data, mask=~mask_zeros, annot=False, linewidths=1.0,
                    linecolor='white', cmap=failed_cmap, cbar=False, ax=ax
                )
                
                ax.set_title(metric_labels.get(metric, metric))
                ax.set_xlabel('') # Clear individual labels for shared axes
                ax.set_ylabel('')

            # Set shared axis labels
            fig.supxlabel("Device Memory (GB)")
            fig.supylabel("Host Memory (GB)")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

            # --- Save the Combined Figure ---
            base_filename = f"{device_name}-{model_size}B-{seq_len}-report"
            pdf_path = os.path.join(pdf_output_dir, f"{base_filename}.pdf")
            plt.savefig(pdf_path)
            png_path = os.path.join(output_dir, f"{base_filename}.png")
            plt.savefig(png_path)
            
            plt.close(fig) # Close the figure to free memory

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