#!/usr/bin/env python3
"""
分析批量测试结果
"""

import pandas as pd

# 读取结果
df = pd.read_csv('samples/detection_results_v20.16_v3.csv')

print("=" * 80)
print("v20.16版本测试结果分析")
print("=" * 80)

# 总体准确率
total = len(df)
correct = len(df[df['is_correct'] == True])
accuracy = correct / total * 100

print(f"\n总体统计：")
print(f"- 总样本数：{total}")
print(f"- 正确识别：{correct}")
print(f"- 错误识别：{total - correct}")
print(f"- 准确率：{accuracy:.2f}%")

# 各类型识别情况
print(f"\n各类型识别情况：")
print("-" * 80)
type_stats = df.groupby('true_label').agg({
    'is_correct': ['count', lambda x: (x == True).sum()]
}).reset_index()
type_stats.columns = ['类型', '总数', '正确数']
type_stats['准确率'] = (type_stats['正确数'] / type_stats['总数'] * 100).round(1)
type_stats['错误数'] = type_stats['总数'] - type_stats['正确数']
print(type_stats.to_string(index=False))

# 识别错误的样本
print(f"\n\n识别错误的样本：")
print("-" * 80)
wrong = df[df['is_correct'] == False]
print(f"共 {len(wrong)} 个样本")
print(f"\n错误类型分布：")
error_distribution = wrong.groupby(['true_label', 'predicted_type']).size().reset_index(name='数量')
error_distribution = error_distribution.sort_values('数量', ascending=False)
print(error_distribution.to_string(index=False))

# 详细错误分析
print(f"\n\n各类型详细错误分析：")
print("-" * 80)
for label in sorted(df['true_label'].unique()):
    label_samples = df[df['true_label'] == label]
    wrong_label_samples = label_samples[label_samples['is_correct'] == False]

    if len(wrong_label_samples) > 0:
        print(f"\n【{label}】识别错误详情（{len(wrong_label_samples)}/{len(label_samples)}）：")
        print("-" * 80)

        # 统计被误判为哪些类型
        predicted_dist = wrong_label_samples['predicted_type'].value_counts()
        print(f"  被误判为：")
        for pred_type, count in predicted_dist.items():
            print(f"    {pred_type}: {count} 个")

        # 显示错误样本的特征
        print(f"\n  错误样本特征（前5个）：")
        for idx, row in wrong_label_samples.head(5).iterrows():
            print(f"    {row['filename'][:50]}...")
            print(f"      段数={row['num_segments']}, 后/前={row['change_last_first']:.2f}, "
                  f"工作状态={row['work_state']}, 前段近零={row['is_first_near_zero']}, "
                  f"后段近零={row['is_last_near_zero']}")
