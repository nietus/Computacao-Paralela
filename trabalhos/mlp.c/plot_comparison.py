#!/usr/bin/env python3
"""
Training Performance Comparison Plotter
Compares CPU, OpenMP CPU, OpenMP GPU, and CUDA implementations
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
LOG_DIR = "./logs"
OUTPUT_FILE = "training_comparison.png"

# Log file paths
LOG_FILES = {
    "Sequential CPU": f"{LOG_DIR}/training_loss_c.txt",
    "OpenMP CPU": f"{LOG_DIR}/training_loss_openmp_cpu.txt",
    "OpenMP GPU": f"{LOG_DIR}/training_loss_openmp_gpu.txt",
    "CUDA GPU": f"{LOG_DIR}/training_loss_cuda.txt",
}

# Colors for each implementation
COLORS = {
    "Sequential CPU": "#1f77b4",      # Blue
    "OpenMP CPU": "#ff7f0e",          # Orange
    "OpenMP GPU": "#2ca02c",          # Green
    "CUDA GPU": "#d62728",            # Red
}

def load_training_data(filepath, max_epochs=10):
    """
    Load training data from CSV file.
    Format: epoch,loss,time
    Returns: epochs, losses, times
    """
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None, None, None

    try:
        data = np.loadtxt(filepath, delimiter=',')
        if data.ndim == 1:  # Single epoch
            data = data.reshape(1, -1)

        # Limit to max_epochs
        if len(data) > max_epochs:
            data = data[:max_epochs]

        epochs = data[:, 0].astype(int)
        losses = data[:, 1]
        times = data[:, 2]
        return epochs, losses, times
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None, None

def main():
    # Load all data
    data = {}
    for name, filepath in LOG_FILES.items():
        epochs, losses, times = load_training_data(filepath)
        if epochs is not None:
            data[name] = {
                'epochs': epochs,
                'losses': losses,
                'times': times,
                'cumulative_time': np.cumsum(times)
            }

    if not data:
        print("Error: No data loaded. Please check log files.")
        return

    # Create figure with subplots (increased spacing)
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35,
                          left=0.08, right=0.95, top=0.93, bottom=0.06)

    # =====================================================================
    # Plot 1: Training Loss vs Epoch
    # =====================================================================
    ax1 = fig.add_subplot(gs[0, :])
    for name, d in data.items():
        ax1.plot(d['epochs'], d['losses'], marker='o', linewidth=2,
                label=name, color=COLORS[name], markersize=6)

    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Average Loss', fontsize=13, fontweight='bold')
    ax1.set_title('Training Loss Comparison Across Implementations',
                  fontsize=15, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)
    ax1.tick_params(labelsize=11)

    # =====================================================================
    # Plot 2: Time per Epoch
    # =====================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    for name, d in data.items():
        ax2.plot(d['epochs'], d['times'], marker='s', linewidth=2,
                label=name, color=COLORS[name], markersize=5)

    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Time per Epoch', fontsize=14, fontweight='bold', pad=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=0)
    ax2.tick_params(labelsize=10)

    # =====================================================================
    # Plot 3: Cumulative Training Time
    # =====================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    for name, d in data.items():
        ax3.plot(d['epochs'], d['cumulative_time'], marker='D', linewidth=2,
                label=name, color=COLORS[name], markersize=5)

    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Time (seconds)', fontsize=12, fontweight='bold')
    ax3.set_title('Cumulative Training Time', fontsize=14, fontweight='bold', pad=12)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(left=0)
    ax3.tick_params(labelsize=10)

    # =====================================================================
    # Plot 4: Average Time per Epoch (Bar Chart)
    # =====================================================================
    ax4 = fig.add_subplot(gs[2, 0])
    names = list(data.keys())
    avg_times = [np.mean(data[name]['times']) for name in names]
    colors_list = [COLORS[name] for name in names]

    bars = ax4.bar(names, avg_times, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, avg_time in zip(bars, avg_times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg_time:.2f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax4.set_ylabel('Average Time (seconds)', fontsize=12, fontweight='bold')
    ax4.set_title('Average Time per Epoch Comparison', fontsize=14, fontweight='bold', pad=12)
    ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax4.tick_params(axis='y', labelsize=10)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=20, ha='right', fontsize=10)

    # =====================================================================
    # Plot 5: Speedup Comparison (relative to Sequential CPU)
    # =====================================================================
    ax5 = fig.add_subplot(gs[2, 1])

    # Calculate speedup relative to Sequential CPU
    if "Sequential CPU" in data:
        baseline_time = np.mean(data["Sequential CPU"]['times'])
        speedups = []
        speedup_names = []
        speedup_colors = []

        for name in names:
            if name != "Sequential CPU":
                avg_time = np.mean(data[name]['times'])
                speedup = baseline_time / avg_time
                speedups.append(speedup)
                speedup_names.append(name)
                speedup_colors.append(COLORS[name])

        bars = ax5.bar(speedup_names, speedups, color=speedup_colors, alpha=0.7,
                      edgecolor='black', linewidth=1.5)

        # Add speedup values on bars
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{speedup:.2f}×',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

        # Add baseline reference line
        ax5.axhline(y=1.0, color='black', linestyle='--', linewidth=2,
                   label='Sequential CPU (baseline)', alpha=0.7)

        ax5.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
        ax5.set_title('Speedup vs Sequential CPU', fontsize=14, fontweight='bold', pad=12)
        ax5.legend(loc='upper right', fontsize=10)
        ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax5.tick_params(axis='y', labelsize=10)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=20, ha='right', fontsize=10)
    else:
        ax5.text(0.5, 0.5, 'Sequential CPU data not available\nfor speedup calculation',
                ha='center', va='center', fontsize=12, transform=ax5.transAxes)

    # =====================================================================
    # Print summary statistics
    # =====================================================================
    print("\n" + "="*70)
    print("TRAINING PERFORMANCE SUMMARY")
    print("="*70)

    for name, d in data.items():
        total_time = np.sum(d['times'])
        avg_time = np.mean(d['times'])
        final_loss = d['losses'][-1]

        print(f"\n{name}:")
        print(f"  Total Training Time: {total_time:.2f} seconds")
        print(f"  Average Time/Epoch:  {avg_time:.2f} seconds")
        print(f"  Final Loss:          {final_loss:.6f}")

        if "Sequential CPU" in data and name != "Sequential CPU":
            baseline_time = np.mean(data["Sequential CPU"]['times'])
            speedup = baseline_time / avg_time
            print(f"  Speedup vs CPU:      {speedup:.2f}×")

    print("\n" + "="*70)

    # =====================================================================
    # Save figure
    # =====================================================================
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison graph saved to: {OUTPUT_FILE}")

    # Show plot
    plt.show()

if __name__ == "__main__":
    main()
