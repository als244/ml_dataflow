import pandas as pd
import argparse
import math
import sys

def create_latex_plot(csv_file_path, seq_train_csv_path=None):
    """
    Generates a LaTeX script for a scatter plot from a given CSV file.

    Args:
        csv_file_path (str): The path to the main input CSV file.
        seq_train_csv_path (str, optional): The path to the sequence train CSV file. Defaults to None.

    Returns:
        str: A string containing the complete LaTeX script.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        return f"Error: The file was not found at '{csv_file_path}'."

    # --- 1. Filter and Prepare Main Data ---
    df_valid = df[df['throughput_tok_per_sec'] != -1.0].copy()
    if df_valid.empty:
        return "Error: No valid data found in the main CSV file (all throughput values are -1)."

    # --- 2. Dynamically Determine Plot Parameters ---
    model_name = df_valid['model_name'].iloc[0]
    seq_len = int(df_valid['seqlen'].iloc[0])
    gbs = int(df_valid['seqs_per_batch'].iloc[0] * df_valid['grad_accum_steps'].iloc[0])

    min_y = math.floor(df_valid['throughput_tok_per_sec'].min() / 500) * 500
    max_y = math.ceil(df_valid['throughput_tok_per_sec'].max() / 500) * 500
    y_ticks = ", ".join(map(str, range(min_y, max_y + 1, 500)))

    x_max = math.ceil(df_valid['peak_device_memory_gb'].max() / 10) * 10
    
    # --- 3. Process Optional Sequence Train Data ---
    seq_train_data_str = ""
    legend_seq_train_entry = ""
    
    if seq_train_csv_path:
        try:
            df_seq = pd.read_csv(seq_train_csv_path)
            if 'device_mem' not in df_seq.columns or 'throughput' not in df_seq.columns:
                 print("Warning: Sequence train CSV is missing 'device_mem' or 'throughput' column. Skipping.", file=sys.stderr)
            else:
                table_content = ["        x y class"]
                for _, row in df_seq.iterrows():
                    table_content.append(f"        {row['device_mem']:.2f} {row['throughput']:.1f} seq")
                
                # FIX: Evaluate the join operation before the f-string
                table_string = "\n".join(table_content)
                seq_train_data_str = (
                    f"    % Sequence Train Data\n"
                    f"    \\addplot[scatter, sharp plot, thin, green!70!black, scatter src=explicit symbolic]\n"
                    f"    table [meta=class] {{\n"
                    f"{table_string}\n"
                    f"    }};"
                )
                
                legend_seq_train_entry = "\\raisebox{{-0.1ex}}{{\\begin{{tikzpicture}}\\path[legend_seq] plot coordinates {{(0,0)}};\\end{{tikzpicture}}}} \\textbf{{Sequence Train}} \\\\[0.2cm]\n"
        except FileNotFoundError:
            print(f"Warning: Sequence train file not found at '{seq_train_csv_path}'. Skipping.", file=sys.stderr)

    # --- 4. Generate Main Plot Data Strings ---
    recompute_map = {0.0: "0", 0.25: "25", 0.50: "50", 0.75: "75", 1.0: "100"}
    color_map = {"0": "color0", "25": "color25", "50": "color50", "75": "color75", "100": "color100"}
    
    plot_data_str = []
    for z_stage in sorted(df_valid['zero_stage'].unique()):
        if z_stage == 0: continue
        plot_data_str.append(f"    % --- ZeRO-{z_stage} Plots ---")
        df_z = df_valid[df_valid['zero_stage'] == z_stage]
        for r_frac, r_str in recompute_map.items():
            subset = df_z[df_z['recompute_frac'] == r_frac]
            if not subset.empty:
                color = color_map[r_str]
                style = f"{color}" if z_stage == 1 else f"draw={color}"
                table_content = ["        x y class"]
                for _, row in subset.iterrows():
                    table_content.append(f"        {row['peak_device_memory_gb']:.2f} {row['throughput_tok_per_sec']:.1f} z{z_stage}_{r_str}")

                # FIX: Evaluate the join operation before the f-string
                table_string = "\n".join(table_content)
                addplot_cmd = (
                    f"    \\addplot[scatter, sharp plot, thin, {style}, scatter src=explicit symbolic] "
                    f"table [meta=class] {{\n"
                    f"{table_string}\n"
                    f"    }};"
                )
                plot_data_str.append(addplot_cmd)
    final_plot_data = "\n".join(plot_data_str)

    # --- 5. Assemble the Final LaTeX Script ---
    latex_template = f"""\
\\documentclass[tikz, border=10pt]{{standalone}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.18}}
\\usetikzlibrary{{shapes.geometric, decorations.pathreplacing}}
\\usepackage{{xcolor}}

% --- Color definitions ---
\\definecolor{{color0}}{{RGB}}{{150, 0, 150}}
\\definecolor{{color25}}{{RGB}}{{27, 85, 146}}
\\definecolor{{color50}}{{RGB}}{{35, 132, 142}}
\\definecolor{{color75}}{{RGB}}{{237, 139, 33}}
\\definecolor{{color100}}{{RGB}}{{212, 59, 56}}

% --- TikZ styles for the custom legend ---
\\tikzset{{
    legend_seq/.style={{mark=diamond*, green!70!black, mark size=3pt}},
    legend_z1_0/.style={{mark=*, color=color0, mark size=3pt}}, legend_z1_25/.style={{mark=*, color=color25, mark size=3pt}}, legend_z1_50/.style={{mark=*, color=color50, mark size=3pt}}, legend_z1_75/.style={{mark=*, color=color75, mark size=3pt}}, legend_z1_100/.style={{mark=*, color=color100, mark size=3pt}},
    legend_z2_0/.style={{mark=square*, mark size=3pt, draw=color0, fill=white, thick}}, legend_z2_25/.style={{mark=square*, mark size=3pt, draw=color25, fill=white, thick}}, legend_z2_50/.style={{mark=square*, mark size=3pt, draw=color50, fill=white, thick}}, legend_z2_75/.style={{mark=square*, mark size=3pt, draw=color75, fill=white, thick}}, legend_z2_100/.style={{mark=square*, mark size=3pt, draw=color100, fill=white, thick}},
    legend_z3_0/.style={{mark=x, draw=color0, mark size=3pt, thick}}, legend_z3_25/.style={{mark=x, draw=color25, mark size=3pt, thick}}, legend_z3_50/.style={{mark=x, draw=color50, mark size=3pt, thick}}, legend_z3_75/.style={{mark=x, draw=color75, mark size=3pt, thick}}, legend_z3_100/.style={{mark=x, draw=color100, mark size=3pt, thick}}
}}

% --- Pgfplots styles ---
\\pgfplotsset{{
    /pgfplots/scatter/classes={{
        seq={{mark=diamond*, color=green!70!black, mark size=4pt}},
        z1_0={{mark=*, color=color0, mark size=3pt}}, z1_25={{mark=*, color=color25, mark size=3pt}}, z1_50={{mark=*, color=color50, mark size=3pt}}, z1_75={{mark=*, color=color75, mark size=3pt}}, z1_100={{mark=*, color=color100, mark size=3pt}},
        z2_0={{mark=square*, mark size=3pt, draw=color0, fill=white, thick}}, z2_25={{mark=square*, mark size=3pt, draw=color25, fill=white, thick}}, z2_50={{mark=square*, mark size=3pt, draw=color50, fill=white, thick}}, z2_75={{mark=square*, mark size=3pt, draw=color75, fill=white, thick}}, z2_100={{mark=square*, mark size=3pt, draw=color100, fill=white, thick}},
        z3_0={{mark=x, mark size=4pt, draw=color0, thick}}, z3_25={{mark=x, mark size=4pt, draw=color25, thick}}, z3_50={{mark=x, mark size=4pt, draw=color50, thick}}, z3_75={{mark=x, mark size=4pt, draw=color75, thick}}, z3_100={{mark=x, mark size=4pt, draw=color100, thick}}
    }}
}}

\\begin{{document}}
\\begin{{tikzpicture}}
    \\begin{{axis}}[
        title={{{model_name} Training Performance on H100 \\\\ \\tiny Full BF16, Seq Length {seq_len}, Seqs per Step={gbs}}},
        title style={{align=center}},
        xlabel={{Peak Device Memory (GiB)}},
        ylabel={{Throughput (Tokens/s)}},
        grid=major,
        grid style={{gray, opacity=0.25}},
        ytick={{{y_ticks}}},
        yticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=0}},
        scaled y ticks=false,
        xmin=0, xmax={x_max},
        xtick={{0, 10, ..., {x_max}}}
    ]

{seq_train_data_str}

{final_plot_data}

    \\end{{axis}}

    % --- Custom Legend ---
    \\node[anchor=north west, draw=black!20, fill=white, inner sep=3pt, font=\\footnotesize]
          at ([xshift=0.3cm,]current axis.north east) {{
        \\begin{{tabular}}{{@{{}}c@{{}}}}
        {legend_seq_train_entry}%
        \\begin{{tabular}}{{@{{,}}l@{{\\quad}}c@{{;}}c@{{;}}c@{{,}}}}
            & \\textbf{{Z1}} & \\textbf{{Z2}} & \\textbf{{Z3}} \\\\[0.05cm]
            \\textbf{{Full Recompute}} & \\raisebox{{-0.1ex}}{{\\begin{{tikzpicture}}\\path[legend_z1_100] plot coordinates {{(0,0)}};\\end{{tikzpicture}}}} & \\raisebox{{-0.1ex}}{{\\begin{{tikzpicture}}\\path[legend_z2_100] plot coordinates {{(0,0)}};\\end{{tikzpicture}}}} & \\raisebox{{-0.1ex}}{{\\begin{{tikzpicture}}\\path[legend_z3_100] plot coordinates {{(0,0)}};\\end{{tikzpicture}}}} \\\\
            \\textbf{{75\\% Recompute}} & \\raisebox{{-0.1ex}}{{\\begin{{tikzpicture}}\\path[legend_z1_75] plot coordinates {{(0,0)}};\\end{{tikzpicture}}}} & \\raisebox{{-0.1ex}}{{\\begin{{tikzpicture}}\\path[legend_z2_75] plot coordinates {{(0,0)}};\\end{{tikzpicture}}}} & \\raisebox{{-0.1ex}}{{\\begin{{tikzpicture}}\\path[legend_z3_75] plot coordinates {{(0,0)}};\\end{{tikzpicture}}}} \\\\[0.05cm]
            \\textbf{{50\\% Recompute}} & \\raisebox{{-0.1ex}}{{\\begin{{tikzpicture}}\\path[legend_z1_50] plot coordinates {{(0,0)}};\\end{{tikzpicture}}}} & \\raisebox{{-0.1ex}}{{\\begin{{tikzpicture}}\\path[legend_z2_50] plot coordinates {{(0,0)}};\\end{{tikzpicture}}}} & \\raisebox{{-0.1ex}}{{\\begin{{tikzpicture}}\\path[legend_z3_50] plot coordinates {{(0,0)}};\\end{{tikzpicture}}}} \\\\[0.05cm]
            \\textbf{{25\\% Recompute}} & \\raisebox{{-0.1ex}}{{\\begin{{tikzpicture}}\\path[legend_z1_25] plot coordinates {{(0,0)}};\\end{{tikzpicture}}}} & \\raisebox{{-0.1ex}}{{\\begin{{tikzpicture}}\\path[legend_z2_25] plot coordinates {{(0,0)}};\\end{{tikzpicture}}}} & \\raisebox{{-0.1ex}}{{\\begin{{tikzpicture}}\\path[legend_z3_25] plot coordinates {{(0,0)}};\\end{{tikzpicture}}}} \\\\[0.05cm]
            \\textbf{{Full Saved}} & \\raisebox{{-0.1ex}}{{\\begin{{tikzpicture}}\\path[legend_z1_0] plot coordinates {{(0,0)}};\\end{{tikzpicture}}}} & \\raisebox{{-0.1ex}}{{\\begin{{tikzpicture}}\\path[legend_z2_0] plot coordinates {{(0,0)}};\\end{{tikzpicture}}}} & \\raisebox{{-0.1ex}}{{\\begin{{tikzpicture}}\\path[legend_z3_0] plot coordinates {{(0,0)}};\\end{{tikzpicture}}}} \\\\
        \\end{{tabular}}
        \\end{{tabular}}
    }};
\\end{{tikzpicture}}
\\end{{document}}
"""
    return latex_template

def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX scatter plot from performance benchmark CSV files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "csv_file",
        help="Path to the main CSV file containing ZeRO benchmark data."
    )
    parser.add_argument(
        "--seqtrain",
        help="Path to the optional CSV file for sequence train data.\nMust contain 'device_mem' and 'throughput' columns.",
        default=None
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to the output .tex file. If not provided, prints to console.",
        default=None
    )
    args = parser.parse_args()

    latex_code = create_latex_plot(args.csv_file, args.seqtrain)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(latex_code)
        print(f"LaTeX script successfully written to {args.output}")
    else:
        print(latex_code)

if __name__ == "__main__":
    main()
