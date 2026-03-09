#!/usr/bin/env python3
"""
批量测试脚本：基于多个样本文件测试故障检测算法
生成详细的测试结果CSV文件
"""

import os
import sys
import csv
from pathlib import Path

# 添加scripts目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from signal_features import load_current_data, extract_features
from detect_fault import classify_fault

def batch_test(sample_dir, output_csv):
    """
    批量测试样本目录中的所有CSV文件
    
    Args:
        sample_dir: 样本文件目录
        output_csv: 输出CSV文件路径
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有CSV文件
    sample_dir_path = Path(sample_dir)
    csv_files = list(sample_dir_path.glob('*.csv'))
    
    if not csv_files:
        print(f"错误：在目录 {sample_dir} 中未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个样本文件")
    
    # 准备结果列表
    results = []
    
    # 处理每个样本文件
    for csv_file in csv_files:
        print(f"处理文件: {csv_file.name}")
        
        try:
            # 加载数据
            signal = load_current_data(str(csv_file))
            
            # 计算特征
            features = extract_features(signal)
            
            # 分类
            result = classify_fault(features)
            
            # 提取关键信息
            file_info = {
                'filename': csv_file.name,
                'predicted_type': result['fault_type'],
                'predicted_type_en': result['fault_type_en'],
                'confidence': result['confidence'],
                'num_segments': features.get('num_segments', 0),
                'work_state': features.get('work_state', 'unknown'),
                'rms_first': features.get('rms_first', 0),
                'rms_mid': features.get('rms_mid', 0),
                'rms_last': features.get('rms_last', 0),
                'change_last_first': features.get('change_last_first', 0),
                'change_mid_first': features.get('change_mid_first', 0),
                'change_last_mid': features.get('change_last_mid', 0),
                'is_first_near_zero': features.get('is_first_near_zero', False),
                'is_last_near_zero': features.get('is_last_near_zero', False),
            }
            
            # 添加各类型评分
            for fault_type, score in result['scores'].items():
                file_info[f'score_{fault_type}'] = score
            
            results.append(file_info)
            
        except Exception as e:
            print(f"  错误: {str(e)}")
            results.append({
                'filename': csv_file.name,
                'error': str(e)
            })
    
    # 写入CSV文件
    if results:
        # 获取所有字段名
        fieldnames = list(results[0].keys())
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n测试完成！结果已保存到: {output_csv}")
        print(f"共处理 {len(results)} 个样本")
    else:
        print("没有生成任何结果")

def main():
    if len(sys.argv) < 2:
        print("用法: python batch_test.py <样本目录> [输出CSV文件]")
        print("示例: python batch_test.py samples/ results.csv")
        sys.exit(1)
    
    sample_dir = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else 'batch_test_results.csv'
    
    if not os.path.isdir(sample_dir):
        print(f"错误：目录 {sample_dir} 不存在")
        sys.exit(1)
    
    batch_test(sample_dir, output_csv)

if __name__ == '__main__':
    main()
