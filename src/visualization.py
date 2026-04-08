"""
Visualization utilities for experiment results
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - prevents plots from displaying
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from .config import RESULTS_DIR, RANDOM_BASELINE_ACCURACY


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                         title="Training Curves", save_path=None):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        title: Plot title
        save_path: Path to save figure (if None, displays only)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[0].set_title('Loss Curves', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracies
    axes[1].plot(epochs, [a*100 for a in train_accs], 'b-o', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, [a*100 for a in val_accs], 'r-s', label='Val Acc', linewidth=2)
    axes[1].axhline(y=RANDOM_BASELINE_ACCURACY*100, color='gray', linestyle='--', 
                    linewidth=2, alpha=0.7, label='Random Baseline')
    axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Accuracy Curves', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
        plt.close()  # Close figure to free memory
    else:
        plt.show()  # Only show if not saving


def plot_comprehensive_comparison(all_results, save_path=None):
    """
    Create comprehensive 6-panel comparison plot for all experiments.
    
    Args:
        all_results: Dict with experiment names as keys and results dict as values
                    Each result should contain: train_loss, val_loss, train_acc, val_acc
        save_path: Path to save figure (if None, displays only)
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.35)
    
    # More distinct colors with better contrast
    colors = ['#2E86AB', '#F77F00', '#06A77D', '#D62828', '#9B59B6']
    markers = ['o', 's', '^', 'D', '*']
    linestyles = ['-', '--', '-.', ':', '-']
    
    # Plot 1: Training Loss
    ax1 = fig.add_subplot(gs[0, 0])
    for idx, (name, data) in enumerate(all_results.items()):
        ax1.plot(range(1, len(data['train_loss'])+1), data['train_loss'], 
                label=name, color=colors[idx], marker=markers[idx], 
                linestyle=linestyles[idx], linewidth=2.5, markersize=8, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_ylim([1.55, 1.70])  # Zoom in to see differences
    
    # Plot 2: Validation Loss
    ax2 = fig.add_subplot(gs[0, 1])
    for idx, (name, data) in enumerate(all_results.items()):
        ax2.plot(range(1, len(data['val_loss'])+1), data['val_loss'], 
                label=name, color=colors[idx], marker=markers[idx],
                linestyle=linestyles[idx], linewidth=2.5, markersize=8, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_ylim([1.60, 1.62])  # Zoom in to see differences
    
    # Plot 3: Training Accuracy
    ax3 = fig.add_subplot(gs[1, 0])
    for idx, (name, data) in enumerate(all_results.items()):
        train_acc_pct = [a * 100 for a in data['train_acc']]
        ax3.plot(range(1, len(train_acc_pct)+1), train_acc_pct, 
                label=name, color=colors[idx], marker=markers[idx],
                linestyle=linestyles[idx], linewidth=2.5, markersize=8, alpha=0.8)
    ax3.axhline(y=RANDOM_BASELINE_ACCURACY*100, color='crimson', linestyle='--', 
                linewidth=2.5, alpha=0.7, label='Random (20%)')
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
    ax3.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.set_ylim([16, 26])  # Focused range
    
    # Plot 4: Validation Accuracy
    ax4 = fig.add_subplot(gs[1, 1])
    for idx, (name, data) in enumerate(all_results.items()):
        val_acc_pct = [a * 100 for a in data['val_acc']]
        ax4.plot(range(1, len(val_acc_pct)+1), val_acc_pct, 
                label=name, color=colors[idx], marker=markers[idx],
                linestyle=linestyles[idx], linewidth=2.5, markersize=8, alpha=0.8)
    ax4.axhline(y=RANDOM_BASELINE_ACCURACY*100, color='crimson', linestyle='--', 
                linewidth=2.5, alpha=0.7, label='Random (20%)')
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
    ax4.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)
    ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax4.set_ylim([16, 24])  # Focused range
    
    # Plot 5: Final Loss Comparison (Overfitting Detection)
    ax5 = fig.add_subplot(gs[2, 0])
    exp_names = [name.split('.')[1].strip() for name in all_results.keys()]
    x_pos = np.arange(len(exp_names))
    train_final = [data['train_loss'][-1] for data in all_results.values()]
    val_final = [data['val_loss'][-1] for data in all_results.values()]
    width = 0.38
    
    bars1 = ax5.bar(x_pos - width/2, train_final, width, label='Train Loss', 
                    color='#4A90E2', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax5.bar(x_pos + width/2, val_final, width, label='Val Loss', 
                    color='#F39C12', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax5.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Final Loss', fontsize=12, fontweight='bold')
    ax5.set_title('Final Loss Comparison (Overfitting Check)', fontsize=14, fontweight='bold', pad=15)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(exp_names, rotation=0, ha='center', fontsize=10)
    ax5.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax5.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    ax5.set_ylim([1.55, 1.65])
    
    # Plot 6: Final Accuracy Bar Chart
    ax6 = fig.add_subplot(gs[2, 1])
    train_acc_final = [data['train_acc'][-1] * 100 for data in all_results.values()]
    val_acc_final = [data['val_acc'][-1] * 100 for data in all_results.values()]
    
    bars1 = ax6.bar(x_pos - width/2, train_acc_final, width, label='Train Acc', 
                    color='#27AE60', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax6.bar(x_pos + width/2, val_acc_final, width, label='Val Acc', 
                    color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1)
    ax6.axhline(y=RANDOM_BASELINE_ACCURACY*100, color='crimson', linestyle='--', 
                linewidth=2.5, alpha=0.7, label='Random (20%)')
    
    ax6.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax6.set_title('Final Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(exp_names, rotation=0, ha='center', fontsize=10)
    ax6.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax6.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    ax6.set_ylim([16, 26])
    
    plt.suptitle('Image Sequence Position Classification - Comprehensive Experiment Analysis',
                fontsize=18, fontweight='bold', y=0.995)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comprehensive plot to {save_path}")
        plt.close()  # Close figure to free memory
    else:
        plt.show()  # Only show if not saving
    
    return fig


def create_results_table(experiments_data):
    """
    Create formatted results table.
    
    Args:
        experiments_data: List of tuples (name, train_loss, val_loss, train_acc, val_acc)
        
    Returns:
        DataFrame with results
    """
    df = pd.DataFrame(experiments_data, columns=[
        'Experiment', 'Train Loss', 'Val Loss', 'Train Acc (%)', 'Val Acc (%)'
    ])
    
    # Convert accuracies to percentages
    df['Train Acc (%)'] = df['Train Acc (%)'] * 100
    df['Val Acc (%)'] = df['Val Acc (%)'] * 100
    
    # Calculate overfitting indicator
    df['Loss Gap'] = df['Val Loss'] - df['Train Loss']
    
    print("\n" + "="*100)
    print("COMPREHENSIVE RESULTS TABLE")
    print("="*100)
    print(df.to_string(index=False, float_format='%.2f'))
    print("="*100)
    print(f"\nRandom Baseline: {RANDOM_BASELINE_ACCURACY*100:.2f}%")
    print(f"Best Model: {df.loc[df['Val Acc (%)'].idxmax(), 'Experiment']} ({df['Val Acc (%)'].max():.2f}%)")
    print(f"Worst Model: {df.loc[df['Val Acc (%)'].idxmin(), 'Experiment']} ({df['Val Acc (%)'].min():.2f}%)")
    print("="*100)
    
    return df


def save_results_table(df, save_path):
    """
    Save results table to CSV.
    
    Args:
        df: Results DataFrame
        save_path: Path to save CSV file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✓ Saved results table to {save_path}")
