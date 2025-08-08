import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from collections import defaultdict

# Set up matplotlib for better visualization
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

def load_evaluation_data(file_path):
    """Load evaluation data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_benchmark_metrics(data, model_name):
    """Extract benchmark metrics from evaluation data"""
    results = data['results']
    benchmarks = []
    
    for task_name, metrics in results.items():
        if 'alias' in metrics and not metrics['alias'].startswith(' - '):
            accuracy = metrics.get('acc,none', 0)
            if accuracy > 0:  # Only include valid data
                benchmarks.append({
                    'benchmark': task_name,
                    'accuracy': accuracy,
                    'model': model_name
                })
    
    return benchmarks

def create_benchmark_comparison_charts():
    """Create comprehensive benchmark comparison charts"""
    print("Loading evaluation data...")
    
    # Load data for both models
    qwen_data = load_evaluation_data('/root/autodl-tmp/minimind/evaluate/qwen/all_2025-08-05T11-08-15.669325.json')
    minimind_data = load_evaluation_data('/root/autodl-tmp/minimind/evaluate/minimind/all_2025-08-05T11-26-51.631302.json')
    
    # Extract metrics
    qwen_benchmarks = extract_benchmark_metrics(qwen_data, 'Qwen2.5-7B')
    minimind_benchmarks = extract_benchmark_metrics(minimind_data, 'MiniMind')
    
    # Create comparison dataframe
    qwen_df = pd.DataFrame(qwen_benchmarks)
    minimind_df = pd.DataFrame(minimind_benchmarks)
    
    # Find common benchmarks
    qwen_benchmarks_set = set(qwen_df['benchmark'])
    minimind_benchmarks_set = set(minimind_df['benchmark'])
    common_benchmarks = qwen_benchmarks_set.intersection(minimind_benchmarks_set)
    
    print(f"Total benchmarks - Qwen: {len(qwen_benchmarks_set)}, MiniMind: {len(minimind_benchmarks_set)}")
    print(f"Common benchmarks: {len(common_benchmarks)}")
    
    # Create comparison data
    comparison_data = []
    for benchmark in common_benchmarks:
        qwen_acc = qwen_df[qwen_df['benchmark'] == benchmark]['accuracy'].iloc[0]
        minimind_acc = minimind_df[minimind_df['benchmark'] == benchmark]['accuracy'].iloc[0]
        comparison_data.append({
            'benchmark': benchmark,
            'qwen_accuracy': qwen_acc,
            'minimind_accuracy': minimind_acc,
            'difference': qwen_acc - minimind_acc,
            'qwen_better': qwen_acc > minimind_acc
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by Qwen performance for better visualization
    comparison_df = comparison_df.sort_values('qwen_accuracy', ascending=False)
    
    # Create comprehensive comparison chart
    create_main_comparison_chart(comparison_df)
    
    # Create category-based analysis
    create_category_analysis(comparison_df)
    
    # Create performance gap analysis
    create_performance_gap_analysis(comparison_df)
    
    # Create statistical summary
    create_statistical_summary(comparison_df)
    
    return comparison_df

def create_main_comparison_chart(comparison_df):
    """Create main benchmark comparison chart"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))
    
    # Main title
    fig.suptitle('Benchmark Performance Comparison: Qwen2.5-7B vs MiniMind', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # Chart 1: Side-by-side comparison
    x = np.arange(len(comparison_df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, comparison_df['qwen_accuracy'], width, 
                    label='Qwen2.5-7B', color='#2E8B57', alpha=0.8)
    bars2 = ax1.bar(x + width/2, comparison_df['minimind_accuracy'], width, 
                    label='MiniMind', color='#FF6347', alpha=0.8)
    
    ax1.set_xlabel('Benchmarks', fontsize=12)
    ax1.set_ylabel('Accuracy Score', fontsize=12)
    ax1.set_title('Accuracy Comparison Across All Benchmarks', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                        for name in comparison_df['benchmark']], 
                       rotation=45, ha='right', fontsize=8)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.005,
                f'{height1:.3f}', ha='center', va='bottom', fontsize=6, rotation=90)
        ax1.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.005,
                f'{height2:.3f}', ha='center', va='bottom', fontsize=6, rotation=90)
    
    # Chart 2: Performance difference
    colors = ['green' if diff > 0 else 'red' for diff in comparison_df['difference']]
    bars = ax2.bar(range(len(comparison_df)), comparison_df['difference'], 
                   color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
    ax2.set_xlabel('Benchmarks', fontsize=12)
    ax2.set_ylabel('Performance Difference (Qwen - MiniMind)', fontsize=12)
    ax2.set_title('Performance Gap Analysis', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(comparison_df)))
    ax2.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                        for name in comparison_df['benchmark']], 
                       rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.005 if height > 0 else -0.015),
                f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=6, rotation=90)
    
    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/benchmark_comparison_main.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Main comparison chart saved as benchmark_comparison_main.png")

def create_category_analysis(comparison_df):
    """Create category-based performance analysis"""
    # Categorize benchmarks by type
    def categorize_benchmark(name):
        name_lower = name.lower()
        if any(keyword in name_lower for keyword in ['math', 'arithmetic', 'algebra']):
            return 'Mathematics'
        elif any(keyword in name_lower for keyword in ['chinese', 'aclue', 'ceval']):
            return 'Chinese Language'
        elif any(keyword in name_lower for keyword in ['cmmlu', 'mmlu']):
            return 'General Knowledge'
        elif any(keyword in name_lower for keyword in ['code', 'programming']):
            return 'Programming'
        elif any(keyword in name_lower for keyword in ['science', 'physics', 'chemistry', 'biology']):
            return 'Science'
        elif any(keyword in name_lower for keyword in ['history', 'geography', 'politics']):
            return 'Social Studies'
        else:
            return 'Other'
    
    comparison_df['category'] = comparison_df['benchmark'].apply(categorize_benchmark)
    
    # Create category summary
    category_summary = comparison_df.groupby('category').agg({
        'qwen_accuracy': ['mean', 'std', 'count'],
        'minimind_accuracy': ['mean', 'std', 'count'],
        'difference': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    category_summary.columns = ['_'.join(col).strip() for col in category_summary.columns]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Category-based Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Average performance by category
    categories = category_summary.index
    qwen_means = category_summary['qwen_accuracy_mean']
    minimind_means = category_summary['minimind_accuracy_mean']
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, qwen_means, width, label='Qwen2.5-7B', 
            color='#2E8B57', alpha=0.8)
    ax1.bar(x + width/2, minimind_means, width, label='MiniMind', 
            color='#FF6347', alpha=0.8)
    
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Average Accuracy')
    ax1.set_title('Average Performance by Category')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance difference by category
    diff_means = category_summary['difference_mean']
    colors = ['green' if diff > 0 else 'red' for diff in diff_means]
    
    ax2.bar(categories, diff_means, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Average Performance Difference')
    ax2.set_title('Performance Gap by Category')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Number of benchmarks per category
    benchmark_counts = category_summary['qwen_accuracy_count']
    ax3.pie(benchmark_counts, labels=categories, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Distribution of Benchmarks by Category')
    
    # 4. Performance variability
    qwen_stds = category_summary['qwen_accuracy_std']
    minimind_stds = category_summary['minimind_accuracy_std']
    
    ax4.bar(x - width/2, qwen_stds, width, label='Qwen2.5-7B', 
            color='#2E8B57', alpha=0.8)
    ax4.bar(x + width/2, minimind_stds, width, label='MiniMind', 
            color='#FF6347', alpha=0.8)
    
    ax4.set_xlabel('Category')
    ax4.set_ylabel('Standard Deviation')
    ax4.set_title('Performance Variability by Category')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/benchmark_category_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Category analysis chart saved as benchmark_category_analysis.png")

def create_performance_gap_analysis(comparison_df):
    """Create detailed performance gap analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Gap Analysis', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: Qwen vs MiniMind performance
    ax1.scatter(comparison_df['qwen_accuracy'], comparison_df['minimind_accuracy'], 
               alpha=0.6, s=50, c='blue')
    
    # Add diagonal line for equal performance
    min_acc = min(comparison_df['qwen_accuracy'].min(), comparison_df['minimind_accuracy'].min())
    max_acc = max(comparison_df['qwen_accuracy'].max(), comparison_df['minimind_accuracy'].max())
    ax1.plot([min_acc, max_acc], [min_acc, max_acc], 'r--', alpha=0.8, label='Equal Performance')
    
    ax1.set_xlabel('Qwen2.5-7B Accuracy')
    ax1.set_ylabel('MiniMind Accuracy')
    ax1.set_title('Performance Correlation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance difference distribution
    ax2.hist(comparison_df['difference'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', label='No Difference')
    ax2.axvline(comparison_df['difference'].mean(), color='green', linestyle='-', 
               label=f'Mean Diff: {comparison_df["difference"].mean():.3f}')
    ax2.set_xlabel('Performance Difference (Qwen - MiniMind)')
    ax2.set_ylabel('Number of Benchmarks')
    ax2.set_title('Distribution of Performance Differences')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Top 10 largest gaps (Qwen advantage)
    top_gaps = comparison_df.nlargest(10, 'difference')
    ax3.barh(range(len(top_gaps)), top_gaps['difference'], color='green', alpha=0.7)
    ax3.set_yticks(range(len(top_gaps)))
    ax3.set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                        for name in top_gaps['benchmark']], fontsize=8)
    ax3.set_xlabel('Performance Advantage (Qwen - MiniMind)')
    ax3.set_title('Top 10 Qwen Advantages')
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance level distribution
    def categorize_performance(acc):
        if acc >= 0.7: return 'Excellent (≥70%)'
        elif acc >= 0.5: return 'Good (50-70%)'
        elif acc >= 0.3: return 'Fair (30-50%)'
        else: return 'Poor (<30%)'
    
    qwen_categories = [categorize_performance(acc) for acc in comparison_df['qwen_accuracy']]
    minimind_categories = [categorize_performance(acc) for acc in comparison_df['minimind_accuracy']]
    
    categories = ['Excellent (≥70%)', 'Good (50-70%)', 'Fair (30-50%)', 'Poor (<30%)']
    qwen_counts = [qwen_categories.count(cat) for cat in categories]
    minimind_counts = [minimind_categories.count(cat) for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax4.bar(x - width/2, qwen_counts, width, label='Qwen2.5-7B', alpha=0.8, color='#2E8B57')
    ax4.bar(x + width/2, minimind_counts, width, label='MiniMind', alpha=0.8, color='#FF6347')
    
    ax4.set_xlabel('Performance Level')
    ax4.set_ylabel('Number of Benchmarks')
    ax4.set_title('Performance Level Distribution')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/benchmark_gap_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Gap analysis chart saved as benchmark_gap_analysis.png")

def create_statistical_summary(comparison_df):
    """Create statistical summary chart"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Statistical Summary of Benchmark Performance', fontsize=16, fontweight='bold')
    
    # 1. Box plot comparison
    data_to_plot = [comparison_df['qwen_accuracy'], comparison_df['minimind_accuracy']]
    box_plot = ax1.boxplot(data_to_plot, labels=['Qwen2.5-7B', 'MiniMind'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('#2E8B57')
    box_plot['boxes'][1].set_facecolor('#FF6347')
    ax1.set_ylabel('Accuracy Score')
    ax1.set_title('Performance Distribution Comparison')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative distribution
    qwen_sorted = np.sort(comparison_df['qwen_accuracy'])
    minimind_sorted = np.sort(comparison_df['minimind_accuracy'])
    y = np.arange(1, len(qwen_sorted) + 1) / len(qwen_sorted)
    
    ax2.plot(qwen_sorted, y, label='Qwen2.5-7B', color='#2E8B57', linewidth=2)
    ax2.plot(minimind_sorted, y, label='MiniMind', color='#FF6347', linewidth=2)
    ax2.set_xlabel('Accuracy Score')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Win/Loss summary
    qwen_wins = sum(comparison_df['qwen_better'])
    minimind_wins = len(comparison_df) - qwen_wins
    
    labels = ['Qwen Wins', 'MiniMind Wins']
    sizes = [qwen_wins, minimind_wins]
    colors = ['#2E8B57', '#FF6347']
    
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title(f'Win/Loss Distribution\n(Total: {len(comparison_df)} benchmarks)')
    
    # 4. Statistical table
    ax4.axis('off')
    
    stats_data = [
        ['Metric', 'Qwen2.5-7B', 'MiniMind', 'Difference'],
        ['Mean', f'{comparison_df["qwen_accuracy"].mean():.4f}', 
         f'{comparison_df["minimind_accuracy"].mean():.4f}', 
         f'{comparison_df["difference"].mean():.4f}'],
        ['Median', f'{comparison_df["qwen_accuracy"].median():.4f}', 
         f'{comparison_df["minimind_accuracy"].median():.4f}', 
         f'{comparison_df["difference"].median():.4f}'],
        ['Std Dev', f'{comparison_df["qwen_accuracy"].std():.4f}', 
         f'{comparison_df["minimind_accuracy"].std():.4f}', 
         f'{comparison_df["difference"].std():.4f}'],
        ['Max', f'{comparison_df["qwen_accuracy"].max():.4f}', 
         f'{comparison_df["minimind_accuracy"].max():.4f}', 
         f'{comparison_df["difference"].max():.4f}'],
        ['Min', f'{comparison_df["qwen_accuracy"].min():.4f}', 
         f'{comparison_df["minimind_accuracy"].min():.4f}', 
         f'{comparison_df["difference"].min():.4f}']
    ]
    
    table = ax4.table(cellText=stats_data[1:], colLabels=stats_data[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(stats_data)):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax4.set_title('Statistical Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/benchmark_statistical_summary.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Statistical summary chart saved as benchmark_statistical_summary.png")

def main():
    """Main function to run benchmark comparison analysis"""
    print("Starting benchmark comparison analysis...")
    print("=" * 50)
    
    # Create all comparison charts
    comparison_df = create_benchmark_comparison_charts()
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("BENCHMARK COMPARISON SUMMARY")
    print("=" * 50)
    print(f"Total benchmarks compared: {len(comparison_df)}")
    print(f"Qwen2.5-7B average accuracy: {comparison_df['qwen_accuracy'].mean():.4f}")
    print(f"MiniMind average accuracy: {comparison_df['minimind_accuracy'].mean():.4f}")
    print(f"Average performance gap: {comparison_df['difference'].mean():.4f}")
    print(f"Qwen wins: {sum(comparison_df['qwen_better'])} out of {len(comparison_df)} benchmarks")
    print(f"Performance gap std dev: {comparison_df['difference'].std():.4f}")
    
    # Top 5 best and worst performance gaps
    print("\nTop 5 Qwen advantages:")
    top_5 = comparison_df.nlargest(5, 'difference')
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"{i}. {row['benchmark'][:30]}... (+{row['difference']:.3f})")
    
    print("\nTop 5 smallest gaps:")
    bottom_5 = comparison_df.nsmallest(5, 'difference')
    for i, (_, row) in enumerate(bottom_5.iterrows(), 1):
        print(f"{i}. {row['benchmark'][:30]}... ({row['difference']:.3f})")
    
    print("\n" + "=" * 50)
    print("Generated files:")
    print("  - benchmark_comparison_main.png: Main comparison chart")
    print("  - benchmark_category_analysis.png: Category-based analysis")
    print("  - benchmark_gap_analysis.png: Performance gap analysis")
    print("  - benchmark_statistical_summary.png: Statistical summary")
    print("=" * 50)
    
    return comparison_df

if __name__ == "__main__":
    main()