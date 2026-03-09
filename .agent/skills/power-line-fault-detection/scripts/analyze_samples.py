#!/usr/bin/env python3
"""
分析指定样本文件
"""

import sys
import json
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from signal_features import load_current_data, extract_features
from detect_fault import classify_fault

files = [
    '../assets/C0322024090002093_2026-02-17 21_20_33.csv',
    '../assets/C0322024090002091_2026-02-17 21_20_34.csv',
    '../assets/C0322024090002093_2026-02-17 21_20_34.csv',
    '../assets/C0322024090002081_2026-02-17 21_20_34.csv',
    '../assets/C0322024090002082_2026-02-17 21_20_34.csv',
    '../assets/C0322024090002092_2026-02-17 21_20_34.csv',
    '../assets/C0322024090002081_2026-02-17 21_20_33.csv',
    '../assets/C0322024090002082_2026-02-17 21_20_33.csv',
    '../assets/C0322024090002092_2026-02-17 21_20_33.csv',
]

print("=" * 100)
print("样本检测结果汇总")
print("=" * 100)

for file_path in files:
    print(f"\n文件: {file_path}")
    print("-" * 100)

    try:
        # 加载数据
        signal = load_current_data(file_path)

        # 计算特征
        features = extract_features(signal)

        # 分类
        result = classify_fault(features)

        # 输出关键信息
        print(f"预测类型: {result['fault_type_cn']} ({result['fault_type']})")
        print(f"置信度: {result['confidence']:.2f}")
        print(f"段数: {features['num_segments']}")
        print(f"工作状态: {features['work_state']}")
        print(f"前段RMS: {features['rms_first']:.2f} (近零: {features['is_first_near_zero']})")
        print(f"后段RMS: {features['rms_last']:.2f} (近零: {features['is_last_near_zero']})")
        print(f"后/前比值: {features['change_last_first']:.2f}")
        print(f"中段RMS: {features['rms_middle']:.2f} (近零: {features['is_middle_near_zero']})")

        # 各类型评分
        print("\n各类型评分:")
        for fault_type, score in sorted(result['scores'].items(), key=lambda x: -x[1]):
            print(f"  {fault_type}: {score:.3f}")

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 100)
