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

def extract_detailed_scores(data, group_name, subtasks):
    """Extract detailed acc_norm scores for a group's subtasks"""
    detailed_scores = []
    for subtask in subtasks:
        if subtask in data['results']:
            score = data['results'][subtask].get('acc_norm,none', None)
            if score is not None:
                detailed_scores.append({
                    'subtask': subtask,
                    'score': score,
                    'group': group_name
                })
    return detailed_scores

def create_detailed_comparison():
    # Load data
    minimind_path = '/root/autodl-tmp/minimind/evaluate/minimind/all_2025-08-05T11-26-51.631302.json'
    qwen_path = '/root/autodl-tmp/minimind/evaluate/qwen/all_2025-08-05T11-08-15.669325.json'
    
    minimind_data = load_evaluation_data(minimind_path)
    qwen_data = load_evaluation_data(qwen_path)
    
    # Get group subtasks
    group_subtasks = minimind_data['group_subtasks']
    main_groups = ['aclue', 'ceval-valid', 'cmmlu', 'tmmluplus']
    
    # Create detailed comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Detailed Performance Analysis by Evaluation Groups', fontsize=16, fontweight='bold')
    
    for idx, group in enumerate(main_groups):
        ax = axes[idx // 2, idx % 2]
        
        if group in group_subtasks:
            subtasks = group_subtasks[group]
            
            # Get scores for both models
            minimind_scores = []
            qwen_scores = []
            task_names = []
            
            for subtask in subtasks:
                if subtask in minimind_data['results'] and subtask in qwen_data['results']:
                    minimind_score = minimind_data['results'][subtask].get('acc_norm,none', None)
                    qwen_score = qwen_data['results'][subtask].get('acc_norm,none', None)
                    
                    if minimind_score is not None and qwen_score is not None:
                        minimind_scores.append(minimind_score)
                        qwen_scores.append(qwen_score)
                        # Simplify task names for better readability
                        task_name = subtask.replace(f'{group}_', '').replace('tmmluplus_', '')
                        task_names.append(task_name)
            
            # Create comparison plot
            x = np.arange(len(task_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, minimind_scores, width, label='MiniMind', 
                          color='#2E86AB', alpha=0.7)
            bars2 = ax.bar(x + width/2, qwen_scores, width, label='Qwen', 
                          color='#A23B72', alpha=0.7)
            
            ax.set_title(f'{group.upper()} - Subtask Performance', fontweight='bold')
            ax.set_ylabel('acc_norm Score')
            ax.set_xlabel('Subtasks')
            ax.set_xticks(x)
            ax.set_xticklabels(task_names, rotation=45, ha='right', fontsize=8)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=6)
            
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=6)
    
    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/minimind/detailed_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create performance gap analysis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    all_gaps = []
    all_groups = []
    
    for group in main_groups:
        if group in group_subtasks:
            subtasks = group_subtasks[group]
            gaps = []
            
            for subtask in subtasks:
                if subtask in minimind_data['results'] and subtask in qwen_data['results']:
                    minimind_score = minimind_data['results'][subtask].get('acc_norm,none', None)
                    qwen_score = qwen_data['results'][subtask].get('acc_norm,none', None)
                    
                    if minimind_score is not None and qwen_score is not None:
                        gap = qwen_score - minimind_score
                        gaps.append(gap)
                        all_gaps.append(gap)
                        all_groups.append(group)
    
    # Create box plot for performance gaps
    gap_df = pd.DataFrame({'Group': all_groups, 'Performance_Gap': all_gaps})
    sns.boxplot(data=gap_df, x='Group', y='Performance_Gap', ax=ax)
    ax.set_title('Performance Gap Distribution (Qwen - MiniMind)', fontweight='bold')
    ax.set_ylabel('Performance Gap (acc_norm)')
    ax.set_xlabel('Evaluation Group')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Gap')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/minimind/performance_gap_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print("\n=== Performance Gap Analysis ===")
    for group in main_groups:
        if group in group_subtasks:
            subtasks = group_subtasks[group]
            gaps = []
            
            for subtask in subtasks:
                if subtask in minimind_data['results'] and subtask in qwen_data['results']:
                    minimind_score = minimind_data['results'][subtask].get('acc_norm,none', None)
                    qwen_score = qwen_data['results'][subtask].get('acc_norm,none', None)
                    
                    if minimind_score is not None and qwen_score is not None:
                        gap = qwen_score - minimind_score
                        gaps.append(gap)
            
            if gaps:
                print(f"\n{group.upper()}:")
                print(f"  Mean Gap: {np.mean(gaps):.4f}")
                print(f"  Std Gap: {np.std(gaps):.4f}")
                print(f"  Min Gap: {np.min(gaps):.4f}")
                print(f"  Max Gap: {np.max(gaps):.4f}")
                print(f"  Tasks where MiniMind performs better: {sum(1 for g in gaps if g < 0)}")
                print(f"  Total tasks: {len(gaps)}")

if __name__ == "__main__":
    create_detailed_comparison()