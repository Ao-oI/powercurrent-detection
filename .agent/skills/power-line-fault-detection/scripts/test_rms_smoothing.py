#!/usr/bin/env python3
"""
测试RMS平滑效果
"""

import numpy as np
from signal_features import compute_rms_aligned

# 创建测试信号：50Hz正弦波 + 两个异常周期
fs = 3200
duration = 0.5  # 0.5秒
n_samples = int(fs * duration)
t = np.linspace(0, duration, n_samples)

# 50Hz正弦波，幅值为7.5
signal = 7.5 * np.sin(2 * np.pi * 50 * t)

# 在周期10和11添加异常值（模拟异常频率信号干扰）
cycle_length = int(fs / 50)  # 64
# 周期10（索引640-703）
signal[640:640+cycle_length] *= 85  # 放大85倍，得到约637.5
# 周期11（索引704-767）
signal[704:704+cycle_length] *= 36  # 放大36倍，得到约270

print("测试信号：")
print(f"- 总长度：{len(signal)} 个采样点")
print(f"- 预期周期数：{len(signal) // cycle_length}")
print(f"- 周期10异常值：幅值约 {7.5 * 85}")
print(f"- 周期11异常值：幅值约 {7.5 * 36}")

# 测试1：不使用平滑
print("\n测试1：不使用RMS平滑（smooth_rms=False）")
rms_unsmoothed = compute_rms_aligned(signal, fs, 50.0, smooth_rms=False)
print(f"RMS序列长度：{len(rms_unsmoothed)}")
print(f"前5个周期RMS：{rms_unsmoothed[:5]}")
print(f"周期8-12 RMS：{rms_unsmoothed[7:12]}")
print(f"后5个周期RMS：{rms_unsmoothed[-5:]}")

# 计算段RMS（模拟3段）
# 前8个周期（周期1-8）
rms_first = np.median(rms_unsmoothed[:8])
# 中间周期（周期9-16）
rms_middle = np.median(rms_unsmoothed[8:16])
# 后面周期（周期17-24）
rms_last = np.median(rms_unsmoothed[16:24])

print(f"\n段RMS（不使用平滑）：")
print(f"- 前段（周期1-8）：{rms_first:.2f}")
print(f"- 中段（周期9-16）：{rms_middle:.2f}")
print(f"- 后段（周期17-24）：{rms_last:.2f}")
print(f"- 后/前比值：{rms_last/rms_first:.2f}")

# 测试2：使用平滑
print("\n测试2：使用RMS平滑（smooth_rms=True）")
rms_smoothed = compute_rms_aligned(signal, fs, 50.0, smooth_rms=True)
print(f"RMS序列长度：{len(rms_smoothed)}")
print(f"前5个周期RMS：{rms_smoothed[:5]}")
print(f"周期8-12 RMS（应过滤异常值）：{rms_smoothed[7:12]}")
print(f"后5个周期RMS：{rms_smoothed[-5:]}")

# 计算段RMS（模拟3段）
rms_first_smooth = np.median(rms_smoothed[:8])
rms_middle_smooth = np.median(rms_smoothed[8:16])
rms_last_smooth = np.median(rms_smoothed[16:24])

print(f"\n段RMS（使用平滑）：")
print(f"- 前段（周期1-8）：{rms_first_smooth:.2f}")
print(f"- 中段（周期9-16）：{rms_middle_smooth:.2f}")
print(f"- 后段（周期17-24）：{rms_last_smooth:.2f}")
print(f"- 后/前比值：{rms_last_smooth/rms_first_smooth:.2f}")

print("\n预期结果：")
print("- 不使用平滑：中段RMS约70-80，后/前比值约1.0（前段也包含异常值）")
print("- 使用平滑：各段RMS约5.3-5.4，后/前比值约1.0")
