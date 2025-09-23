#!/usr/bin/env python3
"""
Script to parse training log files and extract performance metrics into a CSV file.
Parses log files with format: {model_name}_{seqlen}_{seqs_per_batch}_{grad_accum_steps}_{zero_stage}_{save_frac}.log
Extracts throughput, peak device memory, and peak host memory from successful runs.
Failed runs get -1 values for these metrics.
"""

import os
import re
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse filename to extract parameters.
    Expected format: {model_name}_{seqlen}_{seqs_per_batch}_{grad_accum_steps}_{zero_stage}_{save_frac}.log
    
    Returns:
        Dictionary with parsed parameters or None if format doesn't match
    """
    # Remove .log extension
    name_without_ext = filename.replace('.log', '')
    parts = name_without_ext.split('_')
    
    # Should have 6 parts after removing seqs_per_steps (originally 7, now 6)
    if len(parts) < 6:
        print(f"Warning: {filename} doesn't match expected format (has {len(parts)} parts, expected 6)")
        return None
    
    try:
        return {
            'model_name': parts[0],
            'seqlen': parts[1],
            'seqs_per_batch': parts[2],
            'grad_accum_steps': parts[3],
            'zero_stage': parts[4],
            'save_frac': parts[5]
        }
    except (IndexError, ValueError) as e:
        print(f"Error parsing {filename}: {e}")
        return None

def parse_log_file(filepath: Path) -> Tuple[float, float, float]:
    """
    Parse a log file to extract performance metrics.
    
    Returns:
        Tuple of (throughput, peak_device_memory, peak_host_memory)
        Returns (-1, -1, -1) if run failed or metrics not found
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check if run was successful by looking for "Training complete! ✅"
        if "Training complete!" not in content and "Training complete! ✅" not in content:
            return (-1.0, -1.0, -1.0)
        
        # Extract throughput
        throughput_match = re.search(r'Throughput:\s*([\d.]+)\s*Tok/sec', content)
        throughput = float(throughput_match.group(1)) if throughput_match else -1.0
        
        # Extract peak device memory
        device_mem_match = re.search(r'Peak Device Memory Reserved:\s*([\d.]+)\s*GB', content)
        peak_device_mem = float(device_mem_match.group(1)) if device_mem_match else -1.0
        
        # Extract peak host memory
        host_mem_match = re.search(r'Peak Host Memory Reserved:\s*([\d.]+)\s*GB', content)
        peak_host_mem = float(host_mem_match.group(1)) if host_mem_match else -1.0
        
        return (throughput, peak_device_mem, peak_host_mem)
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return (-1.0, -1.0, -1.0)

def process_logs(directory: Path, output_csv: str) -> None:
    """
    Process all .log files in directory and create CSV output.
    """
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return
    
    # Find all .log files
    log_files = list(directory.glob("*.log"))
    
    if not log_files:
        print(f"No .log files found in {directory}")
        return
    
    print(f"Found {len(log_files)} log files in {directory}")
    
    # Process each log file
    results = []
    processed = 0
    
    for log_file in sorted(log_files):
        print(f"Processing {log_file.name}...")
        
        # Parse filename
        params = parse_filename(log_file.name)
        if params is None:
            print(f"Skipping {log_file.name} due to parsing error")
            continue
        
        # Parse log content
        throughput, peak_device_mem, peak_host_mem = parse_log_file(log_file)
        
        # Create result row
        row = {
            'model_name': params['model_name'],
            'seqlen': int(params['seqlen']),
            'seqs_per_batch': int(params['seqs_per_batch']),
            'grad_accum_steps': int(params['grad_accum_steps']),
            'zero_stage': int(params['zero_stage']),
            'recompute_frac': 1 - float(params['save_frac']),
            'throughput_tok_per_sec': throughput,
            'peak_device_memory_gb': peak_device_mem,
            'peak_host_memory_gb': peak_host_mem
        }
        
        results.append(row)
        processed += 1
        
        # Show status
        status = "SUCCESS" if throughput > 0 else "FAILED"
        print(f"  {status}: Throughput={throughput}, Device Mem={peak_device_mem}GB, Host Mem={peak_host_mem}GB")
    
    # Write to CSV
    if results:
        fieldnames = [
            'model_name', 'seqlen', 'seqs_per_batch', 'grad_accum_steps', 
            'zero_stage', 'recompute_frac', 'throughput_tok_per_sec', 
            'peak_device_memory_gb', 'peak_host_memory_gb'
        ]
        
        output_path = Path(output_csv)
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nCSV file created: {output_path}")
        print(f"Processed {processed} files successfully")
        
        # Summary statistics
        successful = sum(1 for r in results if r['throughput_tok_per_sec'] > 0)
        failed = len(results) - successful
        print(f"Successful runs: {successful}")
        print(f"Failed runs: {failed}")
        
        if successful > 0:
            avg_throughput = sum(r['throughput_tok_per_sec'] for r in results if r['throughput_tok_per_sec'] > 0) / successful
            print(f"Average throughput (successful runs): {avg_throughput:.2f} Tok/sec")
    else:
        print("No valid log files were processed")

def main():
    parser = argparse.ArgumentParser(
        description="Parse training log files and extract metrics to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python parse_logs.py dense8B                    # Process logs in dense8B directory
  python parse_logs.py /path/to/logs              # Process logs in specific directory
  python parse_logs.py dense8B -o results.csv    # Custom output filename
        """
    )
    
    parser.add_argument('directory', 
                        help='Directory containing .log files (can be model name or full path)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output CSV filename (default: {directory}_results.csv)')
    
    args = parser.parse_args()
    
    # Handle directory path
    directory = Path(args.directory)
    
    # Generate default output filename if not provided
    if args.output is None:
        output_csv = f"{directory.name}_results.csv"
    else:
        output_csv = args.output
    
    print(f"Input directory: {directory}")
    print(f"Output CSV: {output_csv}")
    print("-" * 50)
    
    process_logs(directory, output_csv)

if __name__ == "__main__":
    main()
