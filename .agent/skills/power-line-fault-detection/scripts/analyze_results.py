#!/usr/bin/env python3
"""
分析测试结果CSV文件，生成统计报告
"""

import csv
import sys
from collections import defaultdict

def analyze_results(csv_file):
    """
    分析测试结果，生成统计报告
    
    Args:
        csv_file: 测试结果CSV文件路径
    """
    # 读取CSV文件
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    print(f"\n{'='*60}")
    print(f"测试结果分析报告")
    print(f"{'='*60}")
    print(f"总样本数: {len(results)}\n")
    
    # 统计各类型的预测数量
    type_counts = defaultdict(int)
    for result in results:
        if 'predicted_type' in result:
            predicted_type = result['predicted_type']
            type_counts[predicted_type] += 1
    
    print(f"预测类型分布:")
    print(f"{'-'*40}")
    for fault_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        percentage = count / len(results) * 100
        print(f"  {fault_type:20s}: {count:3d} ({percentage:5.1f}%)")
    
    # 如果有真实标签（ground_truth），计算准确率
    if 'ground_truth' in results[0]:
        print(f"\n准确率分析:")
        print(f"{'-'*40}")
        
        # 计算总体准确率
        correct = sum(1 for r in results if r['predicted_type'] == r['ground_truth'])
        accuracy = correct / len(results) * 100
        print(f"总体准确率: {correct}/{len(results)} = {accuracy:.2f}%")
        
        # 计算各类型的准确率
        type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        
        for result in results:
            ground_truth = result['ground_truth']
            predicted_type = result['predicted_type']
            type_stats[ground_truth]['total'] += 1
            if ground_truth == predicted_type:
                type_stats[ground_truth]['correct'] += 1
            confusion_matrix[ground_truth][predicted_type] += 1
        
        print(f"\n各类型准确率:")
        print(f"{'-'*40}")
        for fault_type, stats in sorted(type_stats.items()):
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total'] * 100
                print(f"  {fault_type:20s}: {stats['correct']:3d}/{stats['total']:3d} ({accuracy:5.1f}%)")
        
        # 输出混淆矩阵
        print(f"\n混淆矩阵 (行=真实类型, 列=预测类型):")
        print(f"{'-'*80}")
        all_types = sorted(set(list(confusion_matrix.keys()) + 
                            [k for row in confusion_matrix.values() for k in row.keys()]))
        
        # 打印表头
        print(f"{'真实类型':20s}", end='')
        for t in all_types:
            print(f"{t:10s}", end='')
        print()
        
        # 打印每一行
        for real_type in all_types:
            print(f"{real_type:20s}", end='')
            for pred_type in all_types:
                count = confusion_matrix[real_type][pred_type]
                print(f"{count:10d}", end='')
            print()

def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_results.py <测试结果CSV文件>")
        print("示例: python analyze_results.py detection_results_v20.14_detailed.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"错误：文件 {csv_file} 不存在")
        sys.exit(1)
    
    analyze_results(csv_file)

if __name__ == '__main__':
    import os
    main()
