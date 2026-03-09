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


def find_cycle_start(signal: np.ndarray, fs: int = 3200, target_freq: float = 50.0) -> int:
    """
    找到正弦波的起始点（过零点）
    
    通过寻找信号从负变正的过零点，确定周期数据的起始位置
    
    Args:
        signal: 信号数据
        fs: 采样频率
        target_freq: 目标频率（50Hz）
        
    Returns:
        周期起始点的索引
    """
    # 计算理论周期长度（采样点数）
    cycle_length = int(fs / target_freq)  # 3200/50 = 64
    
    # 寻找过零点（从负变正）
    for i in range(1, min(len(signal), cycle_length * 2)):
        if signal[i-1] <= 0 and signal[i] > 0:
            return i
    
    # 如果没有找到过零点，返回0
    return 0


def compute_rms_aligned(signal: np.ndarray, fs: int = 3200, target_freq: float = 50.0, smooth_rms: bool = True) -> np.ndarray:
    """
    计算对齐到周期的RMS序列
    
    v20.15改进：添加RMS序列平滑，过滤异常值（可能由异常频率信号干扰导致）
    
    1. 找到周期起始点（过零点）
    2. 从起始点开始，按周期计算RMS
    3. 使用滑动窗口中值滤波平滑RMS序列
    4. 避免起始点相位误差
    
    Args:
        signal: 信号数据
        fs: 采样频率
        target_freq: 目标频率
        smooth_rms: 是否平滑RMS序列（默认True）
        
    Returns:
        RMS值序列
    """
    # 计算理论周期长度
    cycle_length = int(fs / target_freq)  # 64
    
    # 找到周期起始点
    start_idx = find_cycle_start(signal, fs, target_freq)
    
    # 从起始点开始，按周期计算RMS
    rms_values = []
    for i in range(start_idx, len(signal) - cycle_length + 1, cycle_length):
        cycle = signal[i:i+cycle_length]
        rms = np.sqrt(np.mean(cycle ** 2))
        rms_values.append(rms)
    
    # 如果数据长度不够，使用滑动窗口补充
    if len(rms_values) < 10:
        window_size = cycle_length
        for i in range(len(signal) - window_size + 1):
            window = signal[i:i+window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)
    
    rms_values = np.array(rms_values)
    
    # v20.15：使用滑动窗口中值滤波平滑RMS序列
    # 过滤异常值（可能由异常频率信号干扰导致）
    if smooth_rms and len(rms_values) >= 5:
        window_size = min(5, len(rms_values))
        smoothed_rms = []
        for i in range(len(rms_values)):
            # 获取窗口内的值
            half_window = window_size // 2
            start = max(0, i - half_window)
            end = min(len(rms_values), i + half_window + 1)
            window_values = rms_values[start:end]
            # 使用中值滤波
            smoothed_rms.append(np.median(window_values))
        rms_values = np.array(smoothed_rms)
    
    return rms_values


def detect_segments(rms_values: np.ndarray, 
                     change_threshold: float = 0.3,
                     min_segment_len: int = None) -> tuple:
    """
    自动识别数据段数（v20.4 - 自适应最小段长度）
    
    基于RMS变化率自动识别数据中有多少个稳定段，找出所有突变点
    
    v20.4改进点：
    1. 自适应最小段长度：max(1, len(rms_values)/10)，避免段数过多或过少
    2. 改进突变点合并逻辑：只合并距离非常近的突变点
    3. 改进相似段合并逻辑：使用更严格的相似度判断
    
    Args:
        rms_values: RMS值序列（已对齐到周期）
        change_threshold: 变化率阈值（默认0.3）
        min_segment_len: 最小段长度（默认None，自动计算）
        
    Returns:
        (段数, 各段RMS值列表, 分段索引列表)
    """
    # 自适应最小段长度：至少1个周期，最多占总长度的10%
    if min_segment_len is None:
        min_segment_len = max(1, int(len(rms_values) / 10))
    if len(rms_values) < min_segment_len * 2:
        # 数据量太小，作为1段处理
        return 1, [np.median(rms_values)], [len(rms_values)]
    
    # 计算整体RMS水平，用于自适应阈值
    overall_rms = np.mean(rms_values)
    eps = 1e-6
    
    # 计算RMS变化率（使用滑动窗口）
    window_size = max(1, len(rms_values) // 30)  # 至少1个周期
    change_rates = []
    
    for i in range(window_size, len(rms_values) - window_size):
        prev_mean = np.mean(rms_values[i-window_size:i])
        curr_mean = np.mean(rms_values[i:i+window_size])
        
        # 使用绝对变化量，避免小数值时变化率过大
        abs_change = abs(curr_mean - prev_mean)
        
        # 自适应变化率计算：
        # 如果前后段都非零，使用相对变化率
        # 如果任一段近乎零，使用绝对变化率
        if prev_mean > eps and curr_mean > eps:
            change_rate = abs_change / max(prev_mean, curr_mean)
        else:
            # 对近乎零的段，使用绝对变化量与整体RMS的比值
            change_rate = abs_change / (overall_rms + eps)
        
        change_rates.append(change_rate)
    
    change_rates = np.array(change_rates)
    
    # 找出所有突变点（变化率超过阈值）
    change_indices = np.where(change_rates > change_threshold)[0]
    
    if len(change_indices) == 0:
        # 没有突变点，作为1段处理
        return 1, [np.median(rms_values)], [len(rms_values)]
    
    # 改进的突变点合并逻辑（v20.3）
    # 只合并距离非常近的突变点（<1个周期）
    merged_indices = []
    for idx in change_indices:
        if len(merged_indices) == 0:
            merged_indices.append(idx)
        else:
            # 距离>=1个周期，作为新突变点
            if idx - merged_indices[-1] >= min_segment_len:
                merged_indices.append(idx)
            # 否则，保留变化率更大的那个
            elif change_rates[idx] > change_rates[merged_indices[-1]]:
                merged_indices[-1] = idx
    
    # 确保分段点不在边界（保留至少min_segment_len个周期）
    segment_indices = []
    for idx in merged_indices:
        if idx >= min_segment_len and idx <= len(rms_values) - min_segment_len:
            segment_indices.append(idx)
    
    if len(segment_indices) == 0:
        # 没有合适的分段点，作为1段处理
        return 1, [np.median(rms_values)], [len(rms_values)]
    
    # 计算各段的RMS值
    # v20.15：使用中值而不是平均值，避免异常值影响
    segment_rms_values = []
    prev_idx = 0
    for idx in segment_indices:
        segment_rms = np.median(rms_values[prev_idx:idx])
        segment_rms_values.append(segment_rms)
        prev_idx = idx
    
    # 添加最后一段
    segment_rms = np.median(rms_values[prev_idx:])
    segment_rms_values.append(segment_rms)
    
    # 改进的相似段合并逻辑（v20.5，放宽相似度判断）
    # 只合并确实相似的段（RMS值在0.6-1.67倍范围内，即2/3到3/2）
    merged_segments = [segment_rms_values[0]]
    merged_indices_result = []
    
    if len(segment_indices) > 0:
        merged_indices_result.append(segment_indices[0])
    
    for i in range(1, len(segment_rms_values)):
        prev_rms = segment_rms_values[i-1]
        curr_rms = segment_rms_values[i]
        
        # 放宽相似度判断：0.6-1.67倍范围内（2/3到3/2）
        if 0.6 <= curr_rms / (prev_rms + eps) <= 1.67:
            # 合并：取平均值
            merged_segments[-1] = (merged_segments[-1] + curr_rms) / 2
        else:
            # 不相似，作为新段
            merged_segments.append(curr_rms)
            if i - 1 < len(segment_indices):
                merged_indices_result.append(segment_indices[i-1])
    
    # 确保分段索引和RMS值对应
    if len(merged_segments) > 1:
        segment_indices_final = []
        if len(merged_indices_result) > 0:
            segment_indices_final = merged_indices_result[:-1]
        segment_indices_final.append(len(rms_values))
    else:
        segment_indices_final = [len(rms_values)]
    
    num_segments = len(merged_segments)
    
    return num_segments, merged_segments, segment_indices_final


def extract_time_domain_features(signal: np.ndarray, fs: int = 3200) -> Dict:
    """
    提取时域特征（v20.2 - 改进的科学分段）
    
    改进点：
    1. 使用对齐到周期的RMS计算，避免相位误差
    2. 自适应变化率阈值
    3. 改进正弦比计算（针对非正弦数据）
    
    Args:
        signal: 信号数据
        fs: 采样频率

    Returns:
        时域特征字典
    """
    n = len(signal)

    # 计算RMS序列（对齐到周期，避免相位误差）
    rms_values = compute_rms_aligned(signal, fs, 50.0, smooth_rms=True)
    
    # 如果RMS序列太短，使用滑动窗口补充
    if len(rms_values) < 10:
        window_size = int(fs / 50)  # 64个点
        rms_values = []
        for i in range(len(signal) - window_size + 1):
            window = signal[i:i+window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)
        rms_values = np.array(rms_values)

    # 自动识别段数（v20.2改进版）
    num_segments, segment_rms_values, segment_indices = detect_segments(rms_values)
    
    # 提取最前段和最后段的RMS值
    if num_segments >= 1:
        rms_first = segment_rms_values[0]
    else:
        rms_first = np.median(rms_values)
    
    if num_segments >= 2:
        rms_last = segment_rms_values[-1]
    else:
        rms_last = rms_first
    
    # 提取中段RMS值（如果存在）
    if num_segments >= 3:
        # v20.15：使用中值而不是平均值，避免异常值影响
        rms_middle = np.median(segment_rms_values[1:-1])
    else:
        # 如果只有2段，没有中段，使用前后段的平均值
        rms_middle = (rms_first + rms_last) / 2

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

    # 工作状态判断（基于最前段和最后段）
    if is_first_near_zero and not is_last_near_zero:
        work_state = 'power_on'  # 上电
    elif not is_first_near_zero and is_last_near_zero:
        work_state = 'power_off'  # 停电
    elif 0.5 <= change_last_first <= 2.0:
        work_state = 'normal'  # 正常工作状态
    else:
        work_state = 'unknown'  # 未知

    # 正弦比（std/rms）- 衡量正弦波完美程度
    # 改进：针对非正弦数据，使用分段索引来计算各段的std
    
    # 计算信号采样点的分段边界
    cycle_length = int(fs / 50)  # 64个点
    
    # 前段：从开始到第一个分段点
    first_idx = segment_indices[0] if len(segment_indices) > 0 else len(rms_values)
    first_end = first_idx * cycle_length
    if first_end > 0:
        std_first = np.std(signal[:first_end])
    else:
        std_first = np.std(signal)
    
    # 后段：从最后一个分段点到结束
    last_idx = segment_indices[-1] if len(segment_indices) > 0 else len(rms_values)
    last_start = last_idx * cycle_length
    if last_start < len(signal):
        std_last = np.std(signal[last_start:])
    else:
        std_last = np.std(signal[-cycle_length:]) if len(signal) >= cycle_length else np.std(signal)
    
    if num_segments >= 3:
        # 有中段，计算中段的std
        middle_start = first_end
        middle_end = last_start
        if middle_end > middle_start:
            std_middle = np.std(signal[middle_start:middle_end])
        else:
            std_middle = (std_first + std_last) / 2
    else:
        # 没有明显的中段，使用整体std
        std_middle = (std_first + std_last) / 2

    # 避免除零
    # 对于非正弦数据（如跳闸后段、上电前段），std/rms可能远大于0.707
    # 对于完美正弦波，std/rms = 1/√2 ≈ 0.707
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
        'num_segments': num_segments,  # v20新增：段数
        'segment_rms_values': segment_rms_values,  # v20新增：各段RMS值
        'segment_indices': segment_indices,  # v20新增：分段索引
        'work_state': work_state,  # v20新增：工作状态
    }


def extract_features(signal: np.ndarray, fs: int = 3200) -> Dict:
    """
    提取所有特征（时域 + 频域 + 周期分析）

    Args:
        signal: 信号数据
        fs: 采样频率

    Returns:
        特征字典
    """
    time_features = extract_time_domain_features(signal, fs)
    freq_features = extract_frequency_features(signal, fs)
    
    # v23新增：周期级别波形分析
    cycle_analysis = analyze_beginning_end_cycles(signal, fs, n_cycles=2)

    # 合并特征
    features = {**time_features, **freq_features, **cycle_analysis}

    return features


# ============================================================================
# v23新增：周期级别波形分析（用于扰动检测优化）
# ============================================================================

def extract_cycles_from_segment(segment: np.ndarray, fs: int = 3200, target_freq: float = 50.0) -> list:
    """
    从信号段中提取单个周期
    
    使用过零点检测来识别周期边界
    
    Args:
        segment: 信号段
        fs: 采样频率
        target_freq: 目标频率（50Hz）
        
    Returns:
        周期列表（每个周期是一个数组）
    """
    cycle_length = int(fs / target_freq)  # 64个采样点
    cycles = []
    
    # 寻找第一个过零点（从负变正）
    start_idx = 0
    for i in range(1, min(len(segment), cycle_length * 2)):
        if segment[i-1] <= 0 and segment[i] > 0:
            start_idx = i
            break
    
    # 从过零点开始提取周期
    for i in range(start_idx, len(segment) - cycle_length + 1, cycle_length):
        cycle = segment[i:i+cycle_length]
        if len(cycle) == cycle_length:
            cycles.append(cycle)
    
    return cycles


def calculate_symmetry_score(cycle: np.ndarray) -> float:
    """
    计算周期的对称性分数（0-1）
    
    0 = 完全对称（围绕零对称）
    1 = 完全不对称
    
    Args:
        cycle: 单个周期的数据
        
    Returns:
        对称性分数（0-1）
    """
    if len(cycle) < 2:
        return 0.0
    
    # 找到零点（最接近0的点）
    zero_idx = np.argmin(np.abs(cycle))
    
    # 分割为正半周期和负半周期
    positive_half = cycle[zero_idx:]
    negative_half = cycle[:zero_idx]
    
    if len(positive_half) < 2 or len(negative_half) < 2:
        return 0.0
    
    # 计算峰值对称性
    max_positive = np.max(np.abs(positive_half))
    max_negative = np.max(np.abs(negative_half))
    
    if max_positive + max_negative < 1e-6:
        return 0.0
    
    peak_symmetry = abs(max_positive - max_negative) / (max_positive + max_negative)
    
    # 计算面积对称性
    area_positive = np.sum(np.abs(positive_half))
    area_negative = np.sum(np.abs(negative_half))
    total_area = area_positive + area_negative
    
    if total_area < 1e-6:
        return 0.0
    
    area_symmetry = abs(area_positive - area_negative) / total_area
    
    # 返回平均对称性分数
    symmetry_score = (peak_symmetry + area_symmetry) / 2.0
    
    return min(symmetry_score, 1.0)


def analyze_cycle_waveform(cycle: np.ndarray) -> Dict:
    """
    分析单个周期的波形特性
    
    Args:
        cycle: 单个周期的数据
        
    Returns:
        包含sine_ratio和symmetry_score的字典
    """
    if len(cycle) < 2:
        return {
            'sine_ratio': 0.0,
            'symmetry_score': 0.0,
            'is_non_sinusoidal': True,
            'is_asymmetric': True
        }
    
    # 计算正弦比（std/rms）
    rms = np.sqrt(np.mean(cycle ** 2))
    std = np.std(cycle)
    
    if rms < 1e-6:
        sine_ratio = 0.0
    else:
        sine_ratio = std / rms
    
    # 计算对称性分数
    symmetry_score = calculate_symmetry_score(cycle)
    
    # 判断是否为非正弦波（sine_ratio < 0.95）
    is_non_sinusoidal = sine_ratio < 0.95
    
    # 判断是否为不对称波形（symmetry_score > 0.2）
    is_asymmetric = symmetry_score > 0.2
    
    return {
        'sine_ratio': sine_ratio,
        'symmetry_score': symmetry_score,
        'is_non_sinusoidal': is_non_sinusoidal,
        'is_asymmetric': is_asymmetric
    }


def analyze_beginning_end_cycles(signal: np.ndarray, fs: int = 3200, n_cycles: int = 2) -> Dict:
    """
    提取并分析信号开始和结束的周期
    
    Args:
        signal: 完整信号
        fs: 采样频率
        n_cycles: 要分析的周期数（开始和结束各n个）
        
    Returns:
        包含分析结果的字典
    """
    cycle_length = int(fs / 50.0)  # 64个采样点
    
    # 提取开始周期
    beginning_segment = signal[:min(len(signal), cycle_length * (n_cycles + 2))]
    beginning_cycles = extract_cycles_from_segment(beginning_segment, fs, 50.0)[:n_cycles]
    
    # 提取结束周期
    end_segment = signal[max(0, len(signal) - cycle_length * (n_cycles + 2)):]
    end_cycles = extract_cycles_from_segment(end_segment, fs, 50.0)[-n_cycles:]
    
    # 分析开始周期
    beginning_analysis = []
    for cycle in beginning_cycles:
        analysis = analyze_cycle_waveform(cycle)
        beginning_analysis.append(analysis)
    
    # 分析结束周期
    end_analysis = []
    for cycle in end_cycles:
        analysis = analyze_cycle_waveform(cycle)
        end_analysis.append(analysis)
    
    # 聚合结果
    has_non_sinusoidal_beginning = any(a['is_non_sinusoidal'] for a in beginning_analysis)
    has_non_sinusoidal_end = any(a['is_non_sinusoidal'] for a in end_analysis)
    has_non_sinusoidal_cycles = has_non_sinusoidal_beginning or has_non_sinusoidal_end
    
    has_asymmetric_beginning = any(a['is_asymmetric'] for a in beginning_analysis)
    has_asymmetric_end = any(a['is_asymmetric'] for a in end_analysis)
    has_asymmetric_cycles = has_asymmetric_beginning or has_asymmetric_end
    
    # 计算平均正弦比和对称性分数
    all_sine_ratios = [a['sine_ratio'] for a in beginning_analysis + end_analysis]
    all_symmetry_scores = [a['symmetry_score'] for a in beginning_analysis + end_analysis]
    
    avg_sine_ratio = np.mean(all_sine_ratios) if all_sine_ratios else 1.0
    avg_symmetry_score = np.mean(all_symmetry_scores) if all_symmetry_scores else 0.0
    
    return {
        'beginning_cycles_analysis': beginning_analysis,
        'end_cycles_analysis': end_analysis,
        'has_non_sinusoidal_cycles': has_non_sinusoidal_cycles,
        'has_asymmetric_cycles': has_asymmetric_cycles,
        'has_non_sinusoidal_beginning': has_non_sinusoidal_beginning,
        'has_non_sinusoidal_end': has_non_sinusoidal_end,
        'has_asymmetric_beginning': has_asymmetric_beginning,
        'has_asymmetric_end': has_asymmetric_end,
        'avg_sine_ratio': avg_sine_ratio,
        'avg_symmetry_score': avg_symmetry_score,
        'num_beginning_cycles': len(beginning_cycles),
        'num_end_cycles': len(end_cycles)
    }
