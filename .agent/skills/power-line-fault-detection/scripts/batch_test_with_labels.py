#!/usr/bin/env python3
"""
批量测试脚本：基于所有样本测试故障检测算法
从文件名提取真实标签，对比预测结果
"""

import os
import sys
import csv
import re
from pathlib import Path

# 添加scripts目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.signal_features import load_current_data, extract_features
from scripts.detect_fault import classify_fault

def extract_label_from_filename(filename):
    """
    从文件名提取故障类型标签

    Args:
        filename: 文件名，如 "C0072020010200162_2025-06-30 10_57_05-220kV蒋珍Ⅰ线-跳闸.csv"

    Returns:
        故障类型标签（跳闸、合闸成功、合闸失败、上电、停电、骤降、电流骤升、扰动、未知）
    """
    # 移除.csv后缀
    name = filename.replace('.csv', '')

    # 标准化标签
    label_map = {
        '电流骤增': '电流骤升',
        '电流骤降': '骤降',
        '停电': '骤降',
        '骤降': '骤降',
        '重合闸成功': '合闸成功',
        '合闸成功.': '合闸成功',
    }

    # 查找最后一个"-"之后的内容
    if '-' in name:
        parts = name.split('-')
        label = parts[-1].strip()

        # 检查标签是否有效（排除时间戳如"05 00_19_54"）
        # 有效标签包含中文字符或特定关键词
        if any(ord(c) > 127 for c in label):  # 包含中文字符
            return label_map.get(label, label)
        elif label in label_map:
            return label_map[label]
        else:
            # 可能是时间戳，返回未知
            return '未知'
    else:
        return '未知'

def batch_test(asset_dir, output_csv):
    """
    批量测试样本目录中的所有CSV文件

    Args:
        asset_dir: 样本文件目录
        output_csv: 输出CSV文件路径
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 查找所有CSV文件
    asset_dir_path = Path(asset_dir)
    csv_files = sorted(asset_dir_path.glob('*.csv'))

    if not csv_files:
        print(f"错误：在目录 {asset_dir} 中未找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个样本文件")

    # 准备结果列表
    results = []
    type_counts = {}  # 统计各类型样本数量

    # 处理每个样本文件
    for csv_file in csv_files:
        print(f"处理文件: {csv_file.name}")

        # 提取真实标签
        true_label = extract_label_from_filename(csv_file.name)
        type_counts[true_label] = type_counts.get(true_label, 0) + 1

        try:
            # 加载数据
            signal = load_current_data(str(csv_file))

            # 计算特征
            features = extract_features(signal)

            # 分类
            result = classify_fault(features)

            # 判断是否正确
            predicted_label_en = result['fault_type']
            predicted_label_cn = result['fault_type_cn']

            # 标准化标签映射（英文 -> 中文）
            en_to_cn_map = {
                'disturbance': '扰动',
                'trip': '跳闸',
                'power_off_sag': '停电/骤降',
                'closing_success': '合闸成功',
                'closing_failure': '合闸失败',
                'current_surge': '电流骤升',
                'power_on': '上电',  # v20.16新增
            }
            predicted_label = en_to_cn_map.get(predicted_label_en, predicted_label_cn)

            # 特殊处理：骤降应该匹配"停电/骤降"
            if true_label == '骤降' and predicted_label == '停电/骤降':
                is_correct = True
            else:
                is_correct = (predicted_label == true_label)

            # 提取关键信息
            file_info = {
                'filename': csv_file.name,
                'true_label': true_label,
                'predicted_type': predicted_label,
                'predicted_type_en': predicted_label_en,
                'predicted_type_cn': predicted_label_cn,
                'confidence': result['confidence'],
                'is_correct': is_correct,
                'num_segments': features.get('num_segments', 0),
                'work_state': features.get('work_state', 'unknown'),
                'rms_first': features.get('rms_first', 0),
                'rms_mid': features.get('rms_middle', 0),
                'rms_last': features.get('rms_last', 0),
                'change_last_first': features.get('change_last_first', 0),
                'change_mid_first': features.get('change_mid_first', 0),
                'change_last_mid': features.get('change_last_mid', 0),
                'is_first_near_zero': features.get('is_first_near_zero', False),
                'is_mid_near_zero': features.get('is_middle_near_zero', False),
                'is_last_near_zero': features.get('is_last_near_zero', False),
                'sinusoid_ratio_first': features.get('sinusoid_ratio_first', 0),
                'sinusoid_ratio_last': features.get('sinusoid_ratio_last', 0),
                'signal_length': features.get('signal_length', 0),
            }

            # 添加各类型评分
            for fault_type, score in result['scores'].items():
                file_info[f'score_{fault_type}'] = score

            results.append(file_info)

        except Exception as e:
            print(f"  错误: {str(e)}")
            results.append({
                'filename': csv_file.name,
                'true_label': true_label,
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

        # 打印样本类型分布
        print(f"\n样本类型分布：")
        for label, count in sorted(type_counts.items()):
            print(f"  {label}: {count} 个")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='批量测试故障检测算法')
    parser.add_argument('asset_dir', help='样本文件目录')
    parser.add_argument('output_csv', help='输出CSV文件路径')

    args = parser.parse_args()

    batch_test(args.asset_dir, args.output_csv)
