#!/usr/bin/env python3

import os
import glob
import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("..")
from device_order import *

# Color palette for different GPUs
colors = ["#349999", "#CC1343", "#649903", "#c7aa3e", "#9933CC", "#FF6600", "#0066CC", "#CC6699"]

def load_benchmark_data(filename):
    """Load and parse benchmark data from a txt file."""
    datapoints = [[]]

    try:
        with open(filename, newline="") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)

            for row in csvreader:
                if len(row) == 0:
                    datapoints.append([])
                elif len(row) == 16:
                    # Extract: Arithmetic Intensity, GFlop/s, Power, Clock
                    datapoints[-1].append(
                        [float(row[5]), float(row[9]), float(row[13]), float(row[11])]
                    )

        # Check if we got valid data
        valid_data = [d for d in datapoints if len(d) > 0]
        if len(valid_data) == 0:
            return None

        return datapoints
    except Exception as e:
        print(f"Warning: Could not load {filename}: {e}")
        return None

def plot_single_gpu(filename, datapoints, output_dir="plots"):
    """Generate individual plot for a single GPU."""
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 7))
    gpu_name = os.path.splitext(os.path.basename(filename))[0]

    color = colors[0]

    for i in range(min(1, len(datapoints[1]))):
        ax1.plot(
            [d[i][0] for d in datapoints if len(d) > 0],
            [d[i][1] / 1000 for d in datapoints if len(d) > 0],
            "-o",
            color=color,
            label=gpu_name,
            markersize=3
        )

        ax2.plot(
            [d[i][0] for d in datapoints if len(d) > 0],
            [d[i][2] for d in datapoints if len(d) > 0],
            "-o",
            color=color,
            markersize=3
        )

        ax3.plot(
            [d[i][0] for d in datapoints if len(d) > 0],
            [d[i][3] / 1000 for d in datapoints if len(d) > 0],
            "-o",
            color=color,
            markersize=3
        )

    # Set labels and titles
    ax1.set_title(f"Roofline Analysis - {gpu_name}", fontsize=12, fontweight='bold')
    ax1.legend()
    ax3.set_xlabel("Arithmetic Intensity (Flop/B)")
    ax1.set_ylabel("FP32 (TFlop/s)")
    ax2.set_ylabel("Power (W)")
    ax3.set_ylabel("Clock (GHz)")

    # Set limits
    ax1.set_ylim([0, ax1.get_ylim()[1] * 1.1])
    ax1.set_xlim([0, ax1.get_xlim()[1] * 1.05])
    ax2.set_ylim([0, ax2.get_ylim()[1] * 1.1])
    ax2.set_xlim([0, ax2.get_xlim()[1] * 1.05])
    ax3.set_ylim([0, ax3.get_ylim()[1] * 1.1])
    ax3.set_xlim([0, ax3.get_xlim()[1] * 1.05])

    fig.tight_layout()

    output_file = os.path.join(output_dir, f"{gpu_name}_plot.pdf")
    plt.savefig(output_file, dpi=300)
    print(f"Saved individual plot: {output_file}")
    plt.close()

def plot_comparison(all_data, output_file="plots/comparison_plot.pdf"):
    """Generate comparison plot for all GPUs."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    for idx, (filename, datapoints) in enumerate(all_data.items()):
        gpu_name = os.path.splitext(os.path.basename(filename))[0]
        color = colors[idx % len(colors)]

        for i in range(min(1, len(datapoints[1]))):
            ax1.plot(
                [d[i][0] for d in datapoints if len(d) > 0],
                [d[i][1] / 1000 for d in datapoints if len(d) > 0],
                "-o",
                color=color,
                label=gpu_name,
                markersize=3
            )

            ax2.plot(
                [d[i][0] for d in datapoints if len(d) > 0],
                [d[i][2] for d in datapoints if len(d) > 0],
                "-o",
                color=color,
                markersize=3
            )

            ax3.plot(
                [d[i][0] for d in datapoints if len(d) > 0],
                [d[i][3] / 1000 for d in datapoints if len(d) > 0],
                "-o",
                color=color,
                markersize=3
            )

    # Set labels and titles
    ax1.set_title("GPU Roofline Comparison", fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax3.set_xlabel("Arithmetic Intensity (Flop/B)")
    ax1.set_ylabel("FP32 (TFlop/s)")
    ax2.set_ylabel("Power (W)")
    ax3.set_ylabel("Clock (GHz)")

    # Set limits
    ax1.set_ylim([0, ax1.get_ylim()[1] * 1.1])
    ax1.set_xlim([0, ax1.get_xlim()[1] * 1.05])
    ax2.set_ylim([0, ax2.get_ylim()[1] * 1.1])
    ax2.set_xlim([0, ax2.get_xlim()[1] * 1.05])
    ax3.set_ylim([0, ax3.get_ylim()[1] * 1.1])
    ax3.set_xlim([0, ax3.get_xlim()[1] * 1.05])

    fig.tight_layout()

    plt.savefig(output_file, dpi=300)
    print(f"Saved comparison plot: {output_file}")
    plt.close()

def main():
    print("=" * 60)
    print("GPU Roofline Auto-Plot Tool")
    print("=" * 60)

    # Find all txt files
    txt_files = glob.glob("*.txt")
    print(f"\nFound {len(txt_files)} txt files:")
    for f in txt_files:
        print(f"  - {f}")

    # Load valid benchmark data
    all_data = {}
    print("\nLoading benchmark data...")

    for filename in txt_files:
        datapoints = load_benchmark_data(filename)
        if datapoints is not None:
            all_data[filename] = datapoints
            print(f"{filename}: Valid benchmark data")
        else:
            print(f"{filename}: Invalid or incomplete data (skipped)")

    if len(all_data) == 0:
        print("\n⚠ No valid benchmark data found!")
        print("Please run the benchmark first to generate data files.")
        return

    print(f"\n{len(all_data)} valid benchmark file(s) found.")

    # Generate individual plots
    print("\nGenerating individual plots...")
    for filename, datapoints in all_data.items():
        plot_single_gpu(filename, datapoints)

    # Generate comparison plot
    if len(all_data) > 1:
        print("\nGenerating comparison plot...")
        plot_comparison(all_data)
    else:
        print("\nSkipping comparison plot (only one GPU data available)")

    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
