import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_analyze_data():
    """加载并分析评估数据"""
    file_path = '/root/autodl-tmp/minimind/evaluate/qwen/all_2025-08-05T11-08-15.669325.json'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    
    # 提取主要数据集
    main_datasets = []
    for task_name, metrics in results.items():
        if 'alias' in metrics and not metrics['alias'].startswith(' - '):
            accuracy = metrics.get('acc,none', 0)
            if accuracy > 0:  # 只包含有效数据
                main_datasets.append({
                    'name': task_name,
                    'accuracy': accuracy,
                    'display_name': task_name.upper().replace('-', ' ')
                })
    
    # 按准确率排序
    main_datasets.sort(key=lambda x: x['accuracy'], reverse=True)
    
    return main_datasets, data

def create_comprehensive_summary():
    """创建综合评估总结图表"""
    datasets, raw_data = load_and_analyze_data()
    
    # 创建大图表
    fig = plt.figure(figsize=(16, 12))
    
    # 主标题
    fig.suptitle('MiniMind Model Evaluation Summary\n(Qwen2.5-7B Based)', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # 创建子图布局
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[2, 1], 
                         hspace=0.3, wspace=0.3)
    
    # 1. 主要数据集性能条形图
    ax1 = fig.add_subplot(gs[0, :])
    
    names = [d['display_name'] for d in datasets]
    accuracies = [d['accuracy'] for d in datasets]
    
    # 创建颜色映射
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(accuracies)))
    
    bars = ax1.barh(range(len(names)), accuracies, color=colors)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=10)
    ax1.set_xlabel('Accuracy', fontsize=12)
    ax1.set_title('Performance Across Major Datasets', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{acc:.3f}', va='center', fontsize=9)
    
    # 2. 性能分布直方图
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(accuracies, bins=8, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(accuracies), color='red', linestyle='--', 
                label=f'Mean: {np.mean(accuracies):.3f}')
    ax2.axvline(np.median(accuracies), color='green', linestyle='--', 
                label=f'Median: {np.median(accuracies):.3f}')
    ax2.set_xlabel('Accuracy')
    ax2.set_ylabel('Count')
    ax2.set_title('Accuracy Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 统计信息表格
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    stats_data = [
        ['Metric', 'Value'],
        ['Total Datasets', f'{len(datasets)}'],
        ['Mean Accuracy', f'{np.mean(accuracies):.4f}'],
        ['Std Deviation', f'{np.std(accuracies):.4f}'],
        ['Best Performance', f'{np.max(accuracies):.4f}'],
        ['Worst Performance', f'{np.min(accuracies):.4f}'],
        ['Evaluation Time', '17.4 min']
    ]
    
    table = ax3.table(cellText=stats_data[1:], colLabels=stats_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表格样式
    for i in range(len(stats_data)):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # 标题行
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax3.set_title('Evaluation Statistics', fontweight='bold')
    
    # 4. 性能等级分布饼图
    ax4 = fig.add_subplot(gs[2, 0])
    
    # 定义性能等级
    excellent = sum(1 for acc in accuracies if acc >= 0.6)
    good = sum(1 for acc in accuracies if 0.4 <= acc < 0.6)
    fair = sum(1 for acc in accuracies if 0.25 <= acc < 0.4)
    poor = sum(1 for acc in accuracies if acc < 0.25)
    
    sizes = [excellent, good, fair, poor]
    labels = ['Excellent (≥60%)', 'Good (40-60%)', 'Fair (25-40%)', 'Poor (<25%)']
    colors_pie = ['#2E8B57', '#32CD32', '#FFD700', '#FF6347']
    
    # 只显示非零的部分
    non_zero_sizes = [(size, label, color) for size, label, color in zip(sizes, labels, colors_pie) if size > 0]
    if non_zero_sizes:
        sizes_nz, labels_nz, colors_nz = zip(*non_zero_sizes)
        wedges, texts, autotexts = ax4.pie(sizes_nz, labels=labels_nz, colors=colors_nz, 
                                          autopct='%1.1f%%', startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    ax4.set_title('Performance Level Distribution')
    
    # 5. 顶级和底级表现者
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    top_3 = datasets[:3]
    bottom_3 = datasets[-3:]
    
    text_content = "Top Performers:\n"
    for i, dataset in enumerate(top_3, 1):
        text_content += f"{i}. {dataset['display_name']}: {dataset['accuracy']:.3f}\n"
    
    text_content += "\nBottom Performers:\n"
    for i, dataset in enumerate(bottom_3, 1):
        text_content += f"{i}. {dataset['display_name']}: {dataset['accuracy']:.3f}\n"
    
    ax5.text(0.05, 0.95, text_content, transform=ax5.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/comprehensive_evaluation_summary.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("\n=== MiniMind Model Evaluation Summary ===")
    print(f"Model: Qwen2.5-7B (MiniMind version)")
    print(f"Total datasets evaluated: {len(datasets)}")
    print(f"Average accuracy: {np.mean(accuracies):.4f}")
    print(f"Best performance: {datasets[0]['display_name']} ({datasets[0]['accuracy']:.4f})")
    print(f"Worst performance: {datasets[-1]['display_name']} ({datasets[-1]['accuracy']:.4f})")
    print(f"\nComprehensive summary chart saved as: comprehensive_evaluation_summary.png")

if __name__ == "__main__":
    create_comprehensive_summary()