#!/usr/bin/env python3
"""
信号特征提取模块（v17 - 新增频域特征）
"""

import numpy as np
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


def load_current_data(file_path: str) -> np.ndarray:
    """
    加载电流数据文件

    支持两种格式：
    1. 时间戳 + 电流值：10:52:31.291601428,0.0000000
    2. 仅电流值：12.5

    Args:
        file_path: 文件路径

    Returns:
        电流数据数组
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) == 2:
                # 格式一：时间戳 + 电流值
                try:
                    value = float(parts[1].strip())
                    data.append(value)
                except ValueError:
                    continue
            elif len(parts) == 1:
                # 格式二：仅电流值
                try:
                    value = float(parts[0].strip())
                    data.append(value)
                except ValueError:
                    continue

    return np.array(data)


def extract_frequency_features(signal: np.ndarray, fs: int = 3200) -> Dict:
    """
    提取频域特征（v17新增）

    Args:
        signal: 信号数据
        fs: 采样频率

    Returns:
        频域特征字典
    """
    # FFT变换
    n = len(signal)
    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result)

    # 计算功率谱（归一化）
    power_spectrum = (fft_magnitude ** 2) / n
    total_power = np.sum(power_spectrum)

    # 频率轴
    frequencies = np.fft.fftfreq(n, 1/fs)

    # 1. 工频（50Hz）能量占比
    freq_50Hz_indices = np.where((frequencies >= 45) & (frequencies <= 55))[0]
    if len(freq_50Hz_indices) > 0:
        power_50Hz = np.sum(power_spectrum[freq_50Hz_indices])
        power_ratio_50Hz = power_50Hz / total_power
    else:
        power_ratio_50Hz = 0

    # 2. 频谱集中度（前10个最高频点的能量占比）
    top_k = 10
    top_indices = np.argsort(power_spectrum)[-top_k:]
    power_top = np.sum(power_spectrum[top_indices])
    power_concentration = power_top / total_power

    # 3. 主频能量比（最大频点能量/总能量）
    max_power = np.max(power_spectrum)
    power_ratio_max = max_power / total_power

    # 4. 主频频率
    max_freq_idx = np.argmax(power_spectrum)
    dominant_freq = abs(frequencies[max_freq_idx])

    # 5. 频谱熵（衡量频谱的分散程度）- v17关键特征
    # 归一化功率谱
    ps_norm = power_spectrum / total_power
    # 避免log(0)
    ps_norm = ps_norm[ps_norm > 1e-10]
    spectral_entropy = -np.sum(ps_norm * np.log2(ps_norm))

    return {
        'power_ratio_50Hz': power_ratio_50Hz,
        'power_concentration': power_concentration,
        'power_ratio_max': power_ratio_max,
        'dominant_freq': dominant_freq,
        'spectral_entropy': spectral_entropy
    }


def auto_segment_rms(rms_values: np.ndarray, min_segment_len: int = 64) -> tuple:
    """
    基于RMS变化率自动分段

    找出RMS值发生显著变化的位置（突变点），根据突变点自动划分为前段、中段、后段

    Args:
        rms_values: RMS值序列
        min_segment_len: 最小分段长度

    Returns:
        (前段RMS, 中段RMS, 后段RMS, 分段索引列表)
    """
    eps = 1e-6

    # 如果数据量太小，直接用三等分
    if len(rms_values) < min_segment_len * 3:
        n = len(rms_values)
        segment_len = n // 3
        rms_first = np.mean(rms_values[:segment_len])
        rms_middle = np.mean(rms_values[segment_len:2*segment_len])
        rms_last = np.mean(rms_values[2*segment_len:])
        return rms_first, rms_middle, rms_last, [segment_len, 2*segment_len]

    # 计算RMS变化率（使用滑动窗口）
    window_size = min(32, len(rms_values) // 10)  # 滑动窗口大小
    change_rates = []

    for i in range(window_size, len(rms_values) - window_size):
        prev_mean = np.mean(rms_values[i-window_size:i])
        curr_mean = np.mean(rms_values[i:i+window_size])
        if prev_mean > eps:
            change_rate = abs(curr_mean - prev_mean) / prev_mean
        else:
            change_rate = 0
        change_rates.append(change_rate)

    change_rates = np.array(change_rates)

    # 找出变化率最大的两个点（作为分界点）
    if len(change_rates) == 0:
        # 后备方案：三等分
        n = len(rms_values)
        segment_len = n // 3
        rms_first = np.mean(rms_values[:segment_len])
        rms_middle = np.mean(rms_values[segment_len:2*segment_len])
        rms_last = np.mean(rms_values[2*segment_len:])
        return rms_first, rms_middle, rms_last, [segment_len, 2*segment_len]

    # 找出变化率最大的点（前段到中段的分界）
    # 限制在前1/2范围内
    first_half_indices = np.arange(len(change_rates))[:len(change_rates)//2]
    if len(first_half_indices) > 0:
        first_change_idx = first_half_indices[np.argmax(change_rates[first_half_indices])]
    else:
        first_change_idx = len(change_rates) // 3

    # 找出变化率第二大的点（中段到后段的分界）
    # 限制在first_change_idx之后
    second_half_indices = np.arange(len(change_rates))[first_change_idx+window_size:]
    if len(second_half_indices) > 0:
        second_change_idx = second_half_indices[np.argmax(change_rates[second_half_indices])]
    else:
        second_change_idx = 2 * len(rms_values) // 3

    # 确保分段点顺序正确且有足够的长度
    if second_change_idx <= first_change_idx + min_segment_len:
        second_change_idx = first_change_idx + min_segment_len

    if second_change_idx >= len(rms_values) - min_segment_len:
        second_change_idx = len(rms_values) - min_segment_len - 1

    # 确保每段都有足够的长度
    if first_change_idx < min_segment_len:
        first_change_idx = min_segment_len
    if first_change_idx > len(rms_values) - 2 * min_segment_len:
        first_change_idx = len(rms_values) - 2 * min_segment_len

    # 计算各段的RMS
    rms_first = np.mean(rms_values[:first_change_idx])
    rms_middle = np.mean(rms_values[first_change_idx:second_change_idx])
    rms_last = np.mean(rms_values[second_change_idx:])

    return rms_first, rms_middle, rms_last, [first_change_idx, second_change_idx]


def extract_time_domain_features(signal: np.ndarray, fs: int = 3200) -> Dict:
    """
    提取时域特征（v19 - 自动分段）

    Args:
        signal: 信号数据
        fs: 采样频率

    Returns:
        时域特征字典
    """
    n = len(signal)

    # 计算RMS序列（窗口大小：64个点，对应1个工频周期）
    window_size = 64
    rms_values = []
    for i in range(len(signal) - window_size + 1):
        window = signal[i:i+window_size]
        rms = np.sqrt(np.mean(window ** 2))
        rms_values.append(rms)
    rms_values = np.array(rms_values)

    # 自动分段（v19新增）
    rms_first, rms_middle, rms_last, segment_indices = auto_segment_rms(rms_values)

    # 避免除零
    eps = 1e-6

    # 相对变化率
    change_last_first = rms_last / (rms_first + eps)
    change_mid_first = rms_middle / (rms_first + eps)
    change_last_mid = rms_last / (rms_middle + eps)

    # 整体趋势
    if change_last_first > 1.2:
        overall_trend = 'rising'
    elif change_last_first < 0.8:
        overall_trend = 'falling'
    else:
        overall_trend = 'stable'

    # 近零判断
    near_zero_threshold = 5.0
    is_first_near_zero = rms_first < near_zero_threshold
    is_middle_near_zero = rms_middle < near_zero_threshold
    is_last_near_zero = rms_last < near_zero_threshold

    # 正弦比（std/rms）- 衡量正弦波完美程度
    # 使用自动分段后的索引来计算各段的std
    first_idx, mid_idx = segment_indices
    std_first = np.std(signal[:first_idx*window_size])
    std_middle = np.std(signal[first_idx*window_size:mid_idx*window_size])
    std_last = np.std(signal[mid_idx*window_size:])

    sine_ratio_first = std_first / (rms_first + eps)
    sine_ratio_middle = std_middle / (rms_middle + eps)
    sine_ratio_last = std_last / (rms_last + eps)

    return {
        'rms_first': rms_first,
        'rms_middle': rms_middle,
        'rms_last': rms_last,
        'change_last_first': change_last_first,
        'change_mid_first': change_mid_first,
        'change_last_mid': change_last_mid,
        'overall_trend': overall_trend,
        'is_first_near_zero': is_first_near_zero,
        'is_middle_near_zero': is_middle_near_zero,
        'is_last_near_zero': is_last_near_zero,
        'sine_ratio_first': sine_ratio_first,
        'sine_ratio_middle': sine_ratio_middle,
        'sine_ratio_last': sine_ratio_last,
        'segment_indices': segment_indices,  # v19新增：分段索引
    }


def extract_features(signal: np.ndarray, fs: int = 3200) -> Dict:
    """
    提取所有特征（时域 + 频域）

    Args:
        signal: 信号数据
        fs: 采样频率

    Returns:
        特征字典
    """
    time_features = extract_time_domain_features(signal, fs)
    freq_features = extract_frequency_features(signal, fs)

    # 合并特征
    features = {**time_features, **freq_features}

    return features
