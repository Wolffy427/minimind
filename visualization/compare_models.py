import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from collections import defaultdict
import matplotlib.font_manager as fm

# 解决中文字体问题
def setup_chinese_font():
    """设置中文字体"""
    # 简化字体设置，避免字体查找错误
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("使用默认字体: DejaVu Sans (英文标签)")

def load_evaluation_data(file_path):
    """加载评估数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_main_metrics(data, model_name):
    """提取主要评估指标"""
    results = data['results']
    main_datasets = []
    
    for task_name, metrics in results.items():
        if 'alias' in metrics and not metrics['alias'].startswith(' - '):
            accuracy = metrics.get('acc,none', 0)
            if accuracy > 0:  # 只包含有效数据
                main_datasets.append({
                    'dataset': task_name,
                    'accuracy': accuracy,
                    'model': model_name
                })
    
    return main_datasets

def create_model_comparison():
    """创建模型对比图表"""
    # 设置中文字体
    setup_chinese_font()
    
    # 加载两个模型的数据
    qwen_data = load_evaluation_data('/root/autodl-tmp/minimind/evaluate/qwen/all_2025-08-05T11-08-15.669325.json')
    minimind_data = load_evaluation_data('/root/autodl-tmp/minimind/evaluate/minimind/all_2025-08-05T11-26-51.631302.json')
    
    # 提取指标
    qwen_metrics = extract_main_metrics(qwen_data, 'Qwen2.5-7B')
    minimind_metrics = extract_main_metrics(minimind_data, 'MiniMind')
    
    # 合并数据
    all_metrics = qwen_metrics + minimind_metrics
    df = pd.DataFrame(all_metrics)
    
    # 找到共同的数据集
    qwen_datasets = set([m['dataset'] for m in qwen_metrics])
    minimind_datasets = set([m['dataset'] for m in minimind_metrics])
    common_datasets = qwen_datasets.intersection(minimind_datasets)
    
    print(f"Qwen datasets: {len(qwen_datasets)}")
    print(f"MiniMind datasets: {len(minimind_datasets)}")
    print(f"Common datasets: {len(common_datasets)}")
    
    # 创建对比数据
    comparison_data = []
    for dataset in common_datasets:
        qwen_acc = next(m['accuracy'] for m in qwen_metrics if m['dataset'] == dataset)
        minimind_acc = next(m['accuracy'] for m in minimind_metrics if m['dataset'] == dataset)
        comparison_data.append({
            'dataset': dataset,
            'qwen_accuracy': qwen_acc,
            'minimind_accuracy': minimind_acc,
            'difference': qwen_acc - minimind_acc
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('qwen_accuracy', ascending=False)
    
    # 创建综合对比图
    fig = plt.figure(figsize=(20, 16))
    
    # 主标题
    fig.suptitle('Model Performance Comparison: Qwen2.5-7B vs MiniMind', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # 创建子图布局
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[3, 1], 
                         hspace=0.3, wspace=0.3)
    
    # 1. 主要数据集对比条形图
    ax1 = fig.add_subplot(gs[0, :])
    
    x = np.arange(len(comparison_df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, comparison_df['qwen_accuracy'], width, 
                    label='Qwen2.5-7B', color='#2E8B57', alpha=0.8)
    bars2 = ax1.bar(x + width/2, comparison_df['minimind_accuracy'], width, 
                    label='MiniMind', color='#FF6347', alpha=0.8)
    
    ax1.set_xlabel('Datasets', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Performance Comparison Across Datasets', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df['dataset'], rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    
    # 2. 性能差异图
    ax2 = fig.add_subplot(gs[1, 0])
    
    colors = ['green' if diff > 0 else 'red' for diff in comparison_df['difference']]
    bars = ax2.bar(range(len(comparison_df)), comparison_df['difference'], color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Datasets')
    ax2.set_ylabel('Accuracy Difference (Qwen - MiniMind)')
    ax2.set_title('Performance Gap Analysis')
    ax2.set_xticks(range(len(comparison_df)))
    ax2.set_xticklabels(comparison_df['dataset'], rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. 统计信息表格
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    qwen_avg = comparison_df['qwen_accuracy'].mean()
    minimind_avg = comparison_df['minimind_accuracy'].mean()
    avg_diff = comparison_df['difference'].mean()
    
    stats_data = [
        ['Metric', 'Qwen2.5-7B', 'MiniMind'],
        ['Avg Accuracy', f'{qwen_avg:.4f}', f'{minimind_avg:.4f}'],
        ['Max Accuracy', f'{comparison_df["qwen_accuracy"].max():.4f}', f'{comparison_df["minimind_accuracy"].max():.4f}'],
        ['Min Accuracy', f'{comparison_df["qwen_accuracy"].min():.4f}', f'{comparison_df["minimind_accuracy"].min():.4f}'],
        ['Std Dev', f'{comparison_df["qwen_accuracy"].std():.4f}', f'{comparison_df["minimind_accuracy"].std():.4f}'],
        ['Avg Difference', f'{avg_diff:.4f}', '-']
    ]
    
    table = ax3.table(cellText=stats_data[1:], colLabels=stats_data[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # 设置表格样式
    for i in range(len(stats_data)):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # 标题行
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax3.set_title('Comparison Statistics', fontweight='bold')
    
    # 4. 散点图对比
    ax4 = fig.add_subplot(gs[2, 0])
    
    ax4.scatter(comparison_df['qwen_accuracy'], comparison_df['minimind_accuracy'], 
               alpha=0.6, s=50, c='blue')
    
    # 添加对角线
    min_acc = min(comparison_df['qwen_accuracy'].min(), comparison_df['minimind_accuracy'].min())
    max_acc = max(comparison_df['qwen_accuracy'].max(), comparison_df['minimind_accuracy'].max())
    ax4.plot([min_acc, max_acc], [min_acc, max_acc], 'r--', alpha=0.8, label='Equal Performance')
    
    ax4.set_xlabel('Qwen2.5-7B Accuracy')
    ax4.set_ylabel('MiniMind Accuracy')
    ax4.set_title('Accuracy Correlation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 性能等级分布对比
    ax5 = fig.add_subplot(gs[2, 1])
    
    def categorize_performance(acc):
        if acc >= 0.6: return 'Excellent'
        elif acc >= 0.4: return 'Good'
        elif acc >= 0.25: return 'Fair'
        else: return 'Poor'
    
    qwen_categories = [categorize_performance(acc) for acc in comparison_df['qwen_accuracy']]
    minimind_categories = [categorize_performance(acc) for acc in comparison_df['minimind_accuracy']]
    
    categories = ['Excellent', 'Good', 'Fair', 'Poor']
    qwen_counts = [qwen_categories.count(cat) for cat in categories]
    minimind_counts = [minimind_categories.count(cat) for cat in categories]
    
    x_cat = np.arange(len(categories))
    width = 0.35
    
    ax5.bar(x_cat - width/2, qwen_counts, width, label='Qwen2.5-7B', alpha=0.8)
    ax5.bar(x_cat + width/2, minimind_counts, width, label='MiniMind', alpha=0.8)
    
    ax5.set_xlabel('Performance Level')
    ax5.set_ylabel('Number of Datasets')
    ax5.set_title('Performance Level Distribution')
    ax5.set_xticks(x_cat)
    ax5.set_xticklabels(categories)
    ax5.legend()
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/model_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # 创建详细的数据集对比表
    create_detailed_comparison_table(comparison_df)
    
    # 打印总结
    print("\n=== Model Comparison Summary ===")
    print(f"Qwen2.5-7B average accuracy: {qwen_avg:.4f}")
    print(f"MiniMind average accuracy: {minimind_avg:.4f}")
    print(f"Average performance gap: {avg_diff:.4f}")
    print(f"Qwen wins in {sum(1 for diff in comparison_df['difference'] if diff > 0)} out of {len(comparison_df)} datasets")
    
    return comparison_df

def create_detailed_comparison_table(comparison_df):
    """创建详细的对比表格"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    table_data = []
    table_data.append(['Dataset', 'Qwen2.5-7B', 'MiniMind', 'Difference', 'Winner'])
    
    for _, row in comparison_df.iterrows():
        winner = 'Qwen' if row['difference'] > 0 else 'MiniMind' if row['difference'] < 0 else 'Tie'
        table_data.append([
            row['dataset'][:20] + '...' if len(row['dataset']) > 20 else row['dataset'],
            f"{row['qwen_accuracy']:.4f}",
            f"{row['minimind_accuracy']:.4f}",
            f"{row['difference']:.4f}",
            winner
        ])
    
    # 创建表格
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    # 设置表格样式
    for i in range(len(table_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # 标题行
                cell.set_facecolor('#2E8B57')
                cell.set_text_props(weight='bold', color='white')
            else:
                # 根据胜负设置颜色
                if j == 4:  # Winner列
                    if table_data[i][j] == 'Qwen':
                        cell.set_facecolor('#90EE90')
                    elif table_data[i][j] == 'MiniMind':
                        cell.set_facecolor('#FFB6C1')
                    else:
                        cell.set_facecolor('#FFFFE0')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    plt.title('Detailed Performance Comparison Table', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('/root/autodl-tmp/detailed_comparison_table.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def create_english_summary_chart(comparison_df):
    """创建英文总结图表"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison Analysis Report', fontsize=16, fontweight='bold')
        
        # 1. Average performance comparison
        models = ['Qwen2.5-7B', 'MiniMind']
        avg_scores = [comparison_df['qwen_accuracy'].mean(), comparison_df['minimind_accuracy'].mean()]
        
        bars = ax1.bar(models, avg_scores, color=['#2E8B57', '#FF6347'], alpha=0.8)
        ax1.set_ylabel('Average Accuracy')
        ax1.set_title('Overall Performance Comparison')
        ax1.grid(True, alpha=0.3)
        
        for bar, score in zip(bars, avg_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Win/Loss statistics
        qwen_wins = sum(1 for diff in comparison_df['difference'] if diff > 0)
        minimind_wins = sum(1 for diff in comparison_df['difference'] if diff < 0)
        ties = sum(1 for diff in comparison_df['difference'] if diff == 0)
        
        labels = ['Qwen Wins', 'MiniMind Wins', 'Ties']
        sizes = [qwen_wins, minimind_wins, ties]
        colors = ['#2E8B57', '#FF6347', '#FFD700']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Dataset Win/Loss Distribution')
        
        # 3. Performance difference distribution
        ax3.hist(comparison_df['difference'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(0, color='red', linestyle='--', label='No Difference Line')
        ax3.set_xlabel('Performance Difference (Qwen - MiniMind)')
        ax3.set_ylabel('Number of Datasets')
        ax3.set_title('Performance Difference Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Best and worst performance
        top_5 = comparison_df.nlargest(5, 'qwen_accuracy')
        bottom_5 = comparison_df.nsmallest(5, 'qwen_accuracy')
        
        ax4.axis('off')
        
        text_content = "Top 5 Best Performing Datasets:\n"
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            text_content += f"{i}. {row['dataset'][:20]}...\n   Qwen: {row['qwen_accuracy']:.3f}, MiniMind: {row['minimind_accuracy']:.3f}\n"
        
        text_content += "\nTop 5 Worst Performing Datasets:\n"
        for i, (_, row) in enumerate(bottom_5.iterrows(), 1):
            text_content += f"{i}. {row['dataset'][:20]}...\n   Qwen: {row['qwen_accuracy']:.3f}, MiniMind: {row['minimind_accuracy']:.3f}\n"
        
        ax4.text(0.05, 0.95, text_content, transform=ax4.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('/root/autodl-tmp/english_summary.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print("English summary chart created successfully")
        
    except Exception as e:
        print(f"Error creating summary chart: {e}")

def main():
    """主函数"""
    print("开始创建模型对比可视化...")
    
    # 创建主要对比图表
    comparison_df = create_model_comparison()
    
    # 创建英文总结图表
    create_english_summary_chart(comparison_df)
    
    print("\n所有对比图表已生成完成！")
    print("生成的文件:")
    print("  - model_comparison.png: 综合模型对比分析")
    print("  - detailed_comparison_table.png: 详细对比表格")
    print("  - chinese_summary.png: 中文总结图表（如果字体支持）")

if __name__ == "__main__":
    main()