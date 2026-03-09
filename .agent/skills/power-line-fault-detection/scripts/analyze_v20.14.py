#!/usr/bin/env python3
"""
分析v20.14版本的测试结果
"""

import pandas as pd

# 读取结果
df = pd.read_csv('samples/detection_results_v20.14_detailed.csv')

print("=" * 60)
print("v20.14版本测试结果分析")
print("=" * 60)

# 总体准确率
total = len(df)
correct = len(df[df['是否正确'] == '是'])
accuracy = correct / total * 100

print(f"\n总体统计：")
print(f"- 总样本数：{total}")
print(f"- 正确识别：{correct}")
print(f"- 错误识别：{total - correct}")
print(f"- 准确率：{accuracy:.2f}%")

# 各类型识别情况
print(f"\n各类型识别情况：")
print("-" * 60)
type_stats = df.groupby('期望类型').agg({
    '是否正确': ['count', lambda x: (x == '是').sum()]
}).reset_index()
type_stats.columns = ['类型', '总数', '正确数']
type_stats['准确率'] = (type_stats['正确数'] / type_stats['总数'] * 100).round(1)
type_stats['错误数'] = type_stats['总数'] - type_stats['正确数']
print(type_stats.to_string(index=False))

# 识别错误的样本
print(f"\n识别错误的样本：")
print("-" * 60)
wrong = df[df['是否正确'] == '否']
print(f"共 {len(wrong)} 个样本")
print(f"\n错误类型分布：")
error_distribution = wrong.groupby(['期望类型', '识别类型']).size().reset_index(name='数量')
print(error_distribution.to_string(index=False))

# 问题样本详情
problem_sample = df[df['文件名'] == 'C0322025110004392_2026-02-08 01_01_44.csv']
if len(problem_sample) > 0:
    print(f"\n\n问题样本详情：")
    print("-" * 60)
    print(problem_sample.to_string(index=False))

# 扰动识别情况
print(f"\n\n扰动识别详细分析：")
print("-" * 60)
disturbance_samples = df[df['期望类型'] == '扰动']
if len(disturbance_samples) > 0:
    disturbance_correct = disturbance_samples[disturbance_samples['是否正确'] == '是']
    print(f"扰动样本总数：{len(disturbance_samples)}")
    print(f"扰动识别正确：{len(disturbance_correct)}")
    print(f"扰动识别率：{len(disturbance_correct)/len(disturbance_samples)*100:.1f}%")
    print(f"\n扰动样本详情：")
    print(disturbance_samples[['文件名', '期望类型', '识别类型', '是否正确', '段数', '后/前比值', '工作状态']].to_string(index=False))
