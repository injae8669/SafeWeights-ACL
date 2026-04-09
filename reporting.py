# reporting.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_comprehensive_avg_gradient_plots(avg_gradient_dfs, output_dir, filename_suffix):
    """
    Generate 3x2 comprehensive average gradient charts (Absolute vs Relative).
    """
    print("Generating 3x2 comprehensive average gradient visualization charts...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(3, 2, figsize=(24, 30))
    fig.suptitle('Comprehensive Averaged Gradient Analysis (Absolute vs. Relative)', fontsize=24, y=0.98)

    # --- 1. By Parameter (Top 20) ---
    
    # 1a. Top 20 Absolute Norm (By Parameter)
    ax = axes[0, 0]
    df_params_abs = avg_gradient_dfs['avg_params_abs'].head(20)
    sns.barplot(data=df_params_abs, x='grad_l2_norm', y='param_name', hue='param_name', ax=ax, palette='viridis', legend=False)
    ax.set_title('Top 20 Parameters by Average Absolute Gradient', fontsize=16)
    ax.set_xlabel('Average Gradient L2 Norm', fontsize=12)
    ax.set_ylabel('Parameter Name', fontsize=12)

    # 1b. Top 20 Relative Norm (By Parameter)
    ax = axes[0, 1]
    df_params_rel = avg_gradient_dfs['avg_params_rel'].head(20)
    sns.barplot(data=df_params_rel, x='relative_grad_norm', y='param_name', hue='param_name', ax=ax, palette='rocket', legend=False)
    ax.set_title('Top 20 Parameters by Average Relative Gradient', fontsize=16)
    ax.set_xlabel('Average Relative Gradient Norm', fontsize=12)
    ax.set_ylabel('Parameter Name', fontsize=12)
    
    # --- 2. By Layer ---
    
    # 2a. Absolute Norm Aggregated by Layer
    ax = axes[1, 0]
    df_layers_abs = avg_gradient_dfs['avg_layers_abs'].copy()
    # Sort by layer number to ensure correct plotting order
    df_layers_abs['layer_num'] = df_layers_abs['layer'].str.split('.').str[1].astype(int)
    df_layers_abs = df_layers_abs.sort_values('layer_num')
    sns.barplot(data=df_layers_abs, x='avg_aggregated_abs_norm', y='layer', hue='layer', ax=ax, palette='plasma', legend=False)
    ax.set_title('Average Aggregated Absolute Gradient per Layer', fontsize=16)
    ax.set_xlabel('Average Aggregated Absolute Norm', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)

    # 2b. Relative Norm Aggregated by Layer
    ax = axes[1, 1]
    df_layers_rel = avg_gradient_dfs['avg_layers_rel'].copy()
    # Sort by layer number
    df_layers_rel['layer_num'] = df_layers_rel['layer'].str.split('.').str[1].astype(int)
    df_layers_rel = df_layers_rel.sort_values('layer_num')
    sns.barplot(data=df_layers_rel, x='avg_aggregated_rel_norm', y='layer', hue='layer', ax=ax, palette='coolwarm', legend=False)
    ax.set_title('Average Aggregated Relative Gradient per Layer', fontsize=16)
    ax.set_xlabel('Average Aggregated Relative Norm', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)

    # --- 3. By Component ---
    
    # 3a. Absolute Norm Aggregated by Component
    ax = axes[2, 0]
    df_comp_abs = avg_gradient_dfs['avg_components_abs'].sort_values('avg_aggregated_abs_norm', ascending=False)
    sns.barplot(data=df_comp_abs, x='avg_aggregated_abs_norm', y='component', hue='component', ax=ax, palette='magma', legend=False)
    ax.set_title('Average Aggregated Absolute Gradient per Component', fontsize=16)
    ax.set_xlabel('Average Aggregated Absolute Norm', fontsize=12)
    ax.set_ylabel('Component', fontsize=12)

    # 3b. Relative Norm Aggregated by Component
    ax = axes[2, 1]
    df_comp_rel = avg_gradient_dfs['avg_components_rel'].sort_values('avg_aggregated_rel_norm', ascending=False)
    sns.barplot(data=df_comp_rel, x='avg_aggregated_rel_norm', y='component', hue='component', ax=ax, palette='crest', legend=False)
    ax.set_title('Average Aggregated Relative Gradient per Component', fontsize=16)
    ax.set_xlabel('Average Aggregated Relative Norm', fontsize=12)
    ax.set_ylabel('Component', fontsize=12)

    # --- Save Charts ---
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plot_path = os.path.join(output_dir, f"comprehensive_avg_gradient_plots_{filename_suffix}.png")
    plt.savefig(plot_path)
    print(f"3x2 comprehensive average gradient charts saved to {plot_path}")
    plt.close()


def generate_parameter_jailbreak_summary_plot(rates_abs, rates_rel, output_dir, filename_suffix):
    """
    Generate 2x1 bar chart comparing Jailbreak Success Rate for Top-K and Random perturbations.
    """
    print("Generating Jailbreak Success Rate comparison chart...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('Jailbreak Success Rate Comparison', fontsize=16, y=0.97)

    # --- Chart 1: Analysis Based on Absolute Gradient Norm ---
    data_abs = pd.DataFrame.from_dict(rates_abs, orient='index', columns=['SuccessRate'])
    data_abs.index.name = 'Experiment Type'
    data_abs.reset_index(inplace=True)
    
    sns.barplot(data=data_abs, x='Experiment Type', y='SuccessRate', ax=axes[0], palette='Reds_d')
    axes[0].set_title('Analysis Based on Absolute Gradient Norm')
    axes[0].set_ylabel('Jailbreak Success Rate (Score 4 or 5)')
    axes[0].set_ylim(0, 1.0)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))

    # --- Chart 2: Analysis Based on Relative Gradient Norm ---
    data_rel = pd.DataFrame.from_dict(rates_rel, orient='index', columns=['SuccessRate'])
    data_rel.index.name = 'Experiment Type'
    data_rel.reset_index(inplace=True)
    
    sns.barplot(data=data_rel, x='Experiment Type', y='SuccessRate', ax=axes[1], palette='Blues_d')
    axes[1].set_title('Analysis Based on Relative Gradient Norm')
    axes[1].set_ylabel('Jailbreak Success Rate (Score 4 or 5)')
    axes[1].set_ylim(0, 1.0)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))

    # --- Save Chart ---
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(output_dir, f"parameter_jailbreak_summary_plot_{filename_suffix}.png")
    plt.savefig(plot_path)
    print(f"Jailbreak success rate comparison chart saved to {plot_path}")
    plt.close()

def generate_layer_jailbreak_summary_plot(rates_abs, rates_rel, output_dir, filename_suffix):
    """
    (New) Generate 2x1 bar chart comparing Jailbreak Success Rate for Top-K Layer and Random Layer perturbations.
    """
    print("Generating (Layer) Jailbreak Success Rate comparison chart...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('Jailbreak Success Rate Comparison (Layer Perturbation)', fontsize=16, y=0.97)

    # --- Chart 1: Analysis Based on Absolute Gradient Norm ---
    data_abs = pd.DataFrame.from_dict(rates_abs, orient='index', columns=['SuccessRate'])
    data_abs.index.name = 'Experiment Type'
    data_abs.reset_index(inplace=True)
    
    sns.barplot(data=data_abs, x='Experiment Type', y='SuccessRate', ax=axes[0], palette='Reds_d')
    axes[0].set_title('Analysis Based on Absolute Gradient Norm (Layers)')
    axes[0].set_ylabel('Jailbreak Success Rate (Score 4 or 5)')
    axes[0].set_ylim(0, 1.0)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))

    # --- Chart 2: Analysis Based on Relative Gradient Norm ---
    data_rel = pd.DataFrame.from_dict(rates_rel, orient='index', columns=['SuccessRate'])
    data_rel.index.name = 'Experiment Type'
    data_rel.reset_index(inplace=True)
    
    sns.barplot(data=data_rel, x='Experiment Type', y='SuccessRate', ax=axes[1], palette='Blues_d')
    axes[1].set_title('Analysis Based on Relative Gradient Norm (Layers)')
    axes[1].set_ylabel('Jailbreak Success Rate (Score 4 or 5)')
    axes[1].set_ylim(0, 1.0)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))

    # --- Save Chart ---
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(output_dir, f"layer_jailbreak_summary_plot_{filename_suffix}.png")
    plt.savefig(plot_path)
    print(f"(Layer) Jailbreak success rate comparison chart saved to {plot_path}")
    plt.close()

def generate_component_jailbreak_summary_plot(rates_abs, rates_rel, output_dir, filename_suffix):
    """
    (New) Generate 2x1 bar chart comparing Jailbreak Success Rate for Top-K Component and Random Component perturbations.
    """
    print("Generating (Component) Jailbreak Success Rate comparison chart...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('Jailbreak Success Rate Comparison (Component Perturbation)', fontsize=16, y=0.97)

    # --- Chart 1: Analysis Based on Absolute Gradient Norm ---
    data_abs = pd.DataFrame.from_dict(rates_abs, orient='index', columns=['SuccessRate'])
    data_abs.index.name = 'Experiment Type'
    data_abs.reset_index(inplace=True)
    
    sns.barplot(data=data_abs, x='Experiment Type', y='SuccessRate', ax=axes[0], palette='Reds_d')
    axes[0].set_title('Analysis Based on Absolute Gradient Norm (Components)')
    axes[0].set_ylabel('Jailbreak Success Rate (Score 4 or 5)')
    axes[0].set_ylim(0, 1.0)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))

    # --- Chart 2: Analysis Based on Relative Gradient Norm ---
    data_rel = pd.DataFrame.from_dict(rates_rel, orient='index', columns=['SuccessRate'])
    data_rel.index.name = 'Experiment Type'
    data_rel.reset_index(inplace=True)
    
    sns.barplot(data=data_rel, x='Experiment Type', y='SuccessRate', ax=axes[1], palette='Blues_d')
    axes[1].set_title('Analysis Based on Relative Gradient Norm (Components)')
    axes[1].set_ylabel('Jailbreak Success Rate (Score 4 or 5)')
    axes[1].set_ylim(0, 1.0)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))

    # --- Save Chart ---
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(output_dir, f"component_jailbreak_summary_plot_{filename_suffix}.png")
    plt.savefig(plot_path)
    print(f"(Component) Jailbreak success rate comparison chart saved to {plot_path}")
    plt.close()