import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
import seaborn as sns
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_evaluation_data(file_path):
    """加载评估数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_main_metrics(data):
    """提取主要评估指标"""
    results = data['results']
    main_datasets = []
    sub_tasks = defaultdict(list)
    
    for task_name, metrics in results.items():
        if 'alias' in metrics and not metrics['alias'].startswith(' - '):
            # 主要数据集
            main_datasets.append({
                'dataset': task_name,
                'accuracy': metrics.get('acc,none', 0),
                'acc_norm': metrics.get('acc_norm,none', 0)
            })
        elif 'alias' in metrics and metrics['alias'].startswith(' - '):
            # 子任务
            parent_dataset = task_name.split('_')[0]
            sub_tasks[parent_dataset].append({
                'task': task_name,
                'accuracy': metrics.get('acc,none', 0),
                'acc_norm': metrics.get('acc_norm,none', 0)
            })
    
    return main_datasets, sub_tasks

def create_main_performance_chart(main_datasets):
    """创建主要数据集性能图表"""
    df = pd.DataFrame(main_datasets)
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(df))
    width = 0.35
    
    plt.bar(x - width/2, df['accuracy'], width, label='Accuracy', alpha=0.8)
    plt.bar(x + width/2, df['acc_norm'], width, label='Normalized Accuracy', alpha=0.8)
    
    plt.xlabel('数据集')
    plt.ylabel('准确率')
    plt.title('MiniMind模型在主要评估数据集上的表现')
    plt.xticks(x, df['dataset'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/main_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_heatmap(sub_tasks):
    """创建详细的热力图"""
    # 选择几个主要数据集的子任务
    selected_datasets = ['aclue', 'ceval', 'cmmlu']
    
    for dataset in selected_datasets:
        if dataset in sub_tasks and len(sub_tasks[dataset]) > 5:
            tasks_data = sub_tasks[dataset][:15]  # 取前15个任务
            
            task_names = [task['task'].replace(f'{dataset}_', '') for task in tasks_data]
            accuracies = [task['accuracy'] for task in tasks_data]
            
            plt.figure(figsize=(14, 8))
            colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(accuracies)))
            bars = plt.bar(range(len(task_names)), accuracies, color=colors)
            
            # 添加数值标签
            for i, (bar, acc) in enumerate(zip(bars, accuracies)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.xlabel('子任务')
            plt.ylabel('准确率')
            plt.title(f'{dataset.upper()}数据集各子任务表现详情')
            plt.xticks(range(len(task_names)), task_names, rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(f'/root/autodl-tmp/{dataset}_detailed.png', dpi=300, bbox_inches='tight')
            plt.show()

def create_summary_radar_chart(main_datasets):
    """创建雷达图总结"""
    # 选择主要的几个数据集
    selected_data = [d for d in main_datasets if d['accuracy'] > 0][:8]
    
    if len(selected_data) < 3:
        print("数据不足，无法创建雷达图")
        return
    
    categories = [d['dataset'] for d in selected_data]
    values = [d['accuracy'] for d in selected_data]
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]  # 闭合图形
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
    ax.fill(angles, values, alpha=0.25, color='#1f77b4')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, max(values) * 1.1)
    ax.set_title('MiniMind模型综合表现雷达图', size=16, pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/radar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_distribution(main_datasets):
    """创建性能分布图"""
    accuracies = [d['accuracy'] for d in main_datasets if d['accuracy'] > 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(accuracies, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(accuracies), color='red', linestyle='--', 
                label=f'平均准确率: {np.mean(accuracies):.3f}')
    plt.axvline(np.median(accuracies), color='green', linestyle='--', 
                label=f'中位数准确率: {np.median(accuracies):.3f}')
    
    plt.xlabel('准确率')
    plt.ylabel('数据集数量')
    plt.title('MiniMind模型准确率分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/performance_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_statistics(data, main_datasets):
    """打印总结统计信息"""
    print("\n=== MiniMind模型评估结果总结 ===")
    print(f"模型: {data.get('model_name', 'Unknown')}")
    print(f"评估时间: {data.get('total_evaluation_time_seconds', 'Unknown')}秒")
    
    accuracies = [d['accuracy'] for d in main_datasets if d['accuracy'] > 0]
    if accuracies:
        print(f"\n主要数据集统计:")
        print(f"  平均准确率: {np.mean(accuracies):.4f}")
        print(f"  最高准确率: {np.max(accuracies):.4f}")
        print(f"  最低准确率: {np.min(accuracies):.4f}")
        print(f"  标准差: {np.std(accuracies):.4f}")
        
        print(f"\n表现最好的数据集:")
        best_datasets = sorted(main_datasets, key=lambda x: x['accuracy'], reverse=True)[:5]
        for i, dataset in enumerate(best_datasets, 1):
            print(f"  {i}. {dataset['dataset']}: {dataset['accuracy']:.4f}")

def main():
    """主函数"""
    file_path = '/root/autodl-tmp/minimind/evaluate/qwen/all_2025-08-05T11-08-15.669325.json'
    
    # 加载数据
    print("正在加载评估数据...")
    data = load_evaluation_data(file_path)
    
    # 提取指标
    print("正在提取评估指标...")
    main_datasets, sub_tasks = extract_main_metrics(data)
    
    # 打印统计信息
    print_summary_statistics(data, main_datasets)
    
    # 创建可视化图表
    print("\n正在生成可视化图表...")
    
    # 1. 主要性能图表
    create_main_performance_chart(main_datasets)
    
    # 2. 详细热力图
    create_detailed_heatmap(sub_tasks)
    
    # 3. 雷达图
    create_summary_radar_chart(main_datasets)
    
    # 4. 性能分布图
    create_performance_distribution(main_datasets)
    
    print("\n所有图表已生成完成！")
    print("生成的文件:")
    print("  - main_performance.png: 主要数据集性能对比")
    print("  - aclue_detailed.png: ACLUE数据集详细表现")
    print("  - ceval_detailed.png: C-Eval数据集详细表现")
    print("  - cmmlu_detailed.png: CMMLU数据集详细表现")
    print("  - radar_chart.png: 综合表现雷达图")
    print("  - performance_distribution.png: 准确率分布图")

if __name__ == "__main__":
    main()