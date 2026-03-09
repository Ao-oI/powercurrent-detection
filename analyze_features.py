#!/usr/bin/env python3
import os
import sys

# 分析各类文件的详细特征
folders = {
    'drop_wave': '电流下降',
    'rise_wave': '电流上升',
    'fail_wave': '停电',
    'distur_wave': '扰动',
    'close_fail_wave': '重合闸失败'
}

def rms(v):
    return (sum(x*x for x in v) / len(v)) ** 0.5

def peak(v):
    return max(abs(x) for x in v)

print("=" * 80)
print("各类波形特征详细分析")
print("=" * 80)
print()

for folder, desc in folders.items():
    filepath = f'/home/picohood/projects/local/fl/waves/{folder}/{folder}10.csv'
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            values = [float(line.strip()) for line in f if line.strip()]
        
        n = len(values)
        q_size = n // 4
        
        rms1 = rms(values[:q_size])
        rms2 = rms(values[q_size:2*q_size])
        rms3 = rms(values[2*q_size:3*q_size])
        rms4 = rms(values[3*q_size:])
        
        peak1 = peak(values[:q_size])
        peak2 = peak(values[q_size:2*q_size])
        peak3 = peak(values[2*q_size:3*q_size])
        peak4 = peak(values[3*q_size:])
        
        rms_start = rms1
        rms_end = (rms3 + rms4) / 2
        peak_middle = max(peak2, peak3)
        
        rms_change_ratio = abs(rms_end - rms_start) / (rms_start + 0.1)
        
        smooth1 = abs(rms2 - rms1)
        smooth2 = abs(rms3 - rms2)
        smooth3 = abs(rms4 - rms3)
        is_smooth = smooth1 < 2.5 and smooth2 < 2.5 and smooth3 < 2.5
        
        has_big_peak = peak_middle > rms_start * 1.8
        
        print(f'{desc:12} ({folder}):')
        print(f'  RMS: Q1={rms1:.2f}, Q2={rms2:.2f}, Q3={rms3:.2f}, Q4={rms4:.2f}')
        print(f'  Peak: Q1={peak1:.2f}, Q2={peak2:.2f}, Q3={peak3:.2f}, Q4={peak4:.2f}')
        print(f'  rms_start={rms_start:.2f}, rms_end={rms_end:.2f}')
        print(f'  rms_change_ratio={rms_change_ratio:.2%}')
        print(f'  peak_middle={peak_middle:.2f}, peak/rms={peak_middle/rms_start:.2f}')
        print(f'  平稳性: {smooth1:.2f}, {smooth2:.2f}, {smooth3:.2f} -> {is_smooth}')
        print(f'  hasBigPeak: {has_big_peak}')
        print(f'  rms_start/rms_end={rms_start/rms_end:.2f}' if rms_end > 0.1 else '  rms_end接近0')
        print()
