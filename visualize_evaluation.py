import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

def load_evaluation_data(file_path):
    """Load evaluation data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_group_scores(data, group_name, subtasks):
    """Extract acc_norm scores for a group's subtasks"""
    scores = []
    for subtask in subtasks:
        if subtask in data['results']:
            score = data['results'][subtask].get('acc_norm,none', None)
            if score is not None:
                scores.append(score)
    return scores

def calculate_statistics(scores):
    """Calculate mean, std, min, max for scores"""
    if not scores:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
    
    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores)
    }

def main():
    # Load data
    minimind_path = '/root/autodl-tmp/minimind/evaluate/minimind/all_2025-08-05T11-26-51.631302.json'
    qwen_path = '/root/autodl-tmp/minimind/evaluate/qwen/all_2025-08-05T11-08-15.669325.json'
    
    minimind_data = load_evaluation_data(minimind_path)
    qwen_data = load_evaluation_data(qwen_path)
    
    # Get group subtasks
    group_subtasks = minimind_data['group_subtasks']
    
    # Define main groups (excluding tmmluplus subgroups)
    main_groups = ['aclue', 'ceval-valid', 'cmmlu', 'tmmluplus']
    
    # Collect statistics for both models
    results = []
    
    for group in main_groups:
        if group in group_subtasks:
            subtasks = group_subtasks[group]
            
            # Extract scores for both models
            minimind_scores = extract_group_scores(minimind_data, group, subtasks)
            qwen_scores = extract_group_scores(qwen_data, group, subtasks)
            
            # Calculate statistics
            minimind_stats = calculate_statistics(minimind_scores)
            qwen_stats = calculate_statistics(qwen_scores)
            
            # Store results
            results.append({
                'Group': group,
                'Model': 'MiniMind',
                'Mean': minimind_stats['mean'],
                'Std': minimind_stats['std'],
                'Min': minimind_stats['min'],
                'Max': minimind_stats['max']
            })
            
            results.append({
                'Group': group,
                'Model': 'Qwen',
                'Mean': qwen_stats['mean'],
                'Std': qwen_stats['std'],
                'Min': qwen_stats['min'],
                'Max': qwen_stats['max']
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create visualizations
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison on Evaluation Datasets', fontsize=16, fontweight='bold')
    
    # 1. Mean scores comparison
    ax1 = axes[0, 0]
    mean_data = df.pivot(index='Group', columns='Model', values='Mean')
    mean_data.plot(kind='bar', ax=ax1, color=['#2E86AB', '#A23B72'])
    ax1.set_title('Mean acc_norm Scores by Group', fontweight='bold')
    ax1.set_ylabel('acc_norm Score')
    ax1.set_xlabel('Evaluation Group')
    ax1.legend(title='Model')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Standard deviation comparison
    ax2 = axes[0, 1]
    std_data = df.pivot(index='Group', columns='Model', values='Std')
    std_data.plot(kind='bar', ax=ax2, color=['#2E86AB', '#A23B72'])
    ax2.set_title('Standard Deviation by Group', fontweight='bold')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_xlabel('Evaluation Group')
    ax2.legend(title='Model')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Min-Max range comparison
    ax3 = axes[1, 0]
    groups = df['Group'].unique()
    x = np.arange(len(groups))
    width = 0.35
    
    minimind_df = df[df['Model'] == 'MiniMind']
    qwen_df = df[df['Model'] == 'Qwen']
    
    # Plot min and max as error bars
    ax3.bar(x - width/2, minimind_df['Mean'], width, 
           yerr=[minimind_df['Mean'] - minimind_df['Min'], 
                minimind_df['Max'] - minimind_df['Mean']], 
           label='MiniMind', color='#2E86AB', alpha=0.7, capsize=5)
    
    ax3.bar(x + width/2, qwen_df['Mean'], width, 
           yerr=[qwen_df['Mean'] - qwen_df['Min'], 
                qwen_df['Max'] - qwen_df['Mean']], 
           label='Qwen', color='#A23B72', alpha=0.7, capsize=5)
    
    ax3.set_title('Mean Scores with Min-Max Range', fontweight='bold')
    ax3.set_ylabel('acc_norm Score')
    ax3.set_xlabel('Evaluation Group')
    ax3.set_xticks(x)
    ax3.set_xticklabels(groups, rotation=45)
    ax3.legend(title='Model')
    ax3.grid(True, alpha=0.3)
    
    # 4. Radar chart for mean scores
    ax4 = axes[1, 1]
    ax4.remove()  # Remove the subplot
    ax4 = fig.add_subplot(2, 2, 4, projection='polar')
    
    # Prepare data for radar chart
    angles = np.linspace(0, 2 * np.pi, len(groups), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    minimind_means = minimind_df['Mean'].tolist()
    qwen_means = qwen_df['Mean'].tolist()
    minimind_means += minimind_means[:1]
    qwen_means += qwen_means[:1]
    
    ax4.plot(angles, minimind_means, 'o-', linewidth=2, label='MiniMind', color='#2E86AB')
    ax4.fill(angles, minimind_means, alpha=0.25, color='#2E86AB')
    ax4.plot(angles, qwen_means, 'o-', linewidth=2, label='Qwen', color='#A23B72')
    ax4.fill(angles, qwen_means, alpha=0.25, color='#A23B72')
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(groups)
    ax4.set_title('Mean Performance Radar Chart', fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/minimind/evaluation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n=== Evaluation Results Summary ===")
    print(f"{'Group':<15} {'Model':<10} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 65)
    
    for _, row in df.iterrows():
        print(f"{row['Group']:<15} {row['Model']:<10} {row['Mean']:<8.4f} {row['Std']:<8.4f} {row['Min']:<8.4f} {row['Max']:<8.4f}")
    
    # Calculate overall performance
    print("\n=== Overall Performance ===")
    overall_minimind = df[df['Model'] == 'MiniMind']['Mean'].mean()
    overall_qwen = df[df['Model'] == 'Qwen']['Mean'].mean()
    print(f"MiniMind Overall Mean: {overall_minimind:.4f}")
    print(f"Qwen Overall Mean: {overall_qwen:.4f}")
    print(f"Performance Gap: {overall_qwen - overall_minimind:.4f}")

if __name__ == "__main__":
    main()