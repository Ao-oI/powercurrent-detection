#!/usr/bin/env python3
"""
输电线路工频电流故障识别算法（v22 - 优化合闸失败识别）
基于334个真实样本数据优化

v22核心改进：
1. **优化合闸失败识别**：使用"三段结构+后段衰减"特征
   - 前段近零：front_mean < 15
   - 中段爆发：middle_mean > 50 且 middle/front >= 2.0
   - 后段衰减：back/middle < 0.3（后段明显小于中段）
   - 解决了v21中合闸失败被误判为上电的问题
2. **优化跳闸识别**：放宽后段近零阈值，适应连续衰减
   - 后段近零阈值从15A放宽到30A
   - 支持中段爆发或整体大幅下降两种模式
   - 前段非近零（区别于合闸失败）
3. **调整判断顺序**：优先判断合闸失败，避免被误判为跳闸

关键发现：
- 合闸失败的后段可能有残留电流（几十到几百安培）
- 使用back/middle < 0.3作为合闸失败的关键特征
- 跳闸的rms_last阈值需要放宽，适应连续衰减的数据

继承v21版本的核心改进：
- 优化扰动识别：增强对中间短暂异常值的鲁棒性
- 优化骤升识别：更严格地区分骤升和扰动
- 新增上电识别：区分上电和合闸成功
"""

import numpy as np
from typing import Dict, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from signal_features import load_current_data, extract_features


def calculate_disturbance_score(features: Dict) -> float:
    """
    计算扰动的可能性分数（0-1）- v23优化：添加周期级别波形分析

    v23核心改进：使用周期级别波形分析检测非正弦波和不对称波形
    - 检测开始/结束周期的非正弦波特征（sine_ratio < 0.95）
    - 检测开始/结束周期的不对称特征（symmetry_score > 0.2）
    - 当检测到这些特征时，提升扰动评分

    v21核心改进：区分真实扰动和中间有异常的扰动
    - 真实扰动：前后段幅值相近（change_last_first <= 1.5），中段稳定
    - 中间异常的扰动：前后段幅值相近，但中段有短暂的异常爆发

    扰动特征（v23优化版）：
    1. 前后段相近（后/前比值 <= 1.5，更严格）
    2. 中段稳定性（允许短暂异常，但不能持续爆发）
    3. 非近零（前后段都不是近零状态）
    4. 频域特征：频谱集中度高（>0.7）
    5. 周期分析：开始/结束周期非正弦波或不对称（v23新增）

    关键区别：
    - 扰动：前后段幅值相近，整体稳定，允许短暂中间异常，可能有非正弦波特征
    - 跳闸：中段爆发，后段衰减
    - 骤升：后段持续高于前段（后/前 > 1.5）
    """
    change_last_first = features['change_last_first']
    change_mid_first = features['change_mid_first']
    change_last_mid = features['change_last_mid']
    is_first_near_zero = features['is_first_near_zero']
    is_last_near_zero = features['is_last_near_zero']
    power_concentration = features['power_concentration']
    spectral_entropy = features['spectral_entropy']
    num_segments = features.get('num_segments', 3)
    
    # v23新增：周期分析特征
    has_non_sinusoidal_cycles = features.get('has_non_sinusoidal_cycles', False)
    has_asymmetric_cycles = features.get('has_asymmetric_cycles', False)

    score = 0.0

    # 特征1：前后段相近（后/前比值 <= 1.5，更严格）
    # 这是扰动的核心特征：前后段工作状态相近
    # v21优化：使用更严格的1.5阈值，区分扰动和骤升
    if 0.8 <= change_last_first <= 1.2:
        score += 0.5  # 非常接近
    elif 0.7 <= change_last_first <= 1.3:
        score += 0.4  # 很接近
    elif change_last_first <= 1.5:
        score += 0.3  # 相近（允许一定差异）
    elif change_last_first > 1.5 and change_last_first <= 2.0:
        score += 0.1  # 轻微差异，可能是扰动
    else:
        score -= 0.5  # 前后段差异太大，不是扰动

    # 特征2：中段稳定性（允许短暂异常）
    # v21优化：放宽中段稳定性要求，允许短暂异常，但不能持续爆发
    # 中段爆发是跳闸的关键特征，所以中段不能持续爆发
    if 0.7 <= change_mid_first <= 1.3 and 0.7 <= change_last_mid <= 1.3:
        score += 0.4  # 非常稳定
    elif 0.5 <= change_mid_first <= 2.0 and 0.5 <= change_last_mid <= 2.0:
        score += 0.3  # 稳定
    elif change_mid_first > 3.0 or change_last_mid > 3.0:
        score -= 1.0  # 中段持续爆发，这是跳闸的特征，不是扰动
    # 允许中段有短暂异常（1.5-3.0），轻微扣分但不完全排除
    elif 2.0 < change_mid_first <= 3.0 or 2.0 < change_last_mid <= 3.0:
        score -= 0.2  # 中段有短暂异常，可能是扰动，但置信度降低

    # 特征3：非近零（前后段都不是近零状态）
    # 扰动是正常工作状态的波动，不是近零状态
    if not is_first_near_zero and not is_last_near_zero:
        score += 0.2
    elif is_first_near_zero or is_last_near_zero:
        score -= 0.3  # 近零状态，不是扰动

    # 特征4：频域特征 - 辅助特征
    # 扰动频谱集中度高（>0.7），故障频谱分散
    if power_concentration > 0.9:
        score += 0.1
    elif power_concentration > 0.7:
        score += 0.05

    # 特征5：频谱熵 - 辅助特征
    # 扰动频谱熵低（<4.5），故障频谱熵高
    if spectral_entropy < 3.5:
        score += 0.1
    elif spectral_entropy < 4.5:
        score += 0.05
    
    # 特征6：v23新增 - 周期级别波形分析
    # 扰动可能在开始/结束周期有非正弦波或不对称特征
    # 这是区分扰动和其他故障类型的关键特征
    if has_non_sinusoidal_cycles or has_asymmetric_cycles:
        # 如果前后段比值接近（<=1.5），这强烈表明是扰动
        if change_last_first <= 1.5:
            score += 0.4  # 大幅提升评分
        else:
            # 即使前后段比值>1.5，如果有非正弦波/不对称特征，也可能是扰动
            score += 0.2

    return max(0.0, min(score, 1.0))


def calculate_trip_score(features: Dict) -> float:
    """计算跳闸的可能性分数（0-1）"""
    rms_first = features['rms_first']
    rms_middle = features['rms_middle']
    rms_last = features['rms_last']
    change_mid_first = features['change_mid_first']
    change_last_mid = features['change_last_mid']
    is_first_near_zero = features['is_first_near_zero']
    is_last_near_zero = features['is_last_near_zero']

    score = 0.0

    # 特征1：前段非近零（区别于合闸）
    if not is_first_near_zero and rms_first > 5.0:
        score += 0.3

    # 特征2：中段爆发强度
    if change_mid_first > 2.0:
        score += 0.5
    elif change_mid_first > 1.5:
        score += 0.3

    # 特征3：后段近零或明显下降（关键特征）
    if is_last_near_zero:
        score += 0.6
    elif rms_last < 5.0:
        score += 0.4

    # 特征4：后段衰减
    if change_last_mid < 0.5:
        score += 0.2

    return min(score, 1.0)


def calculate_power_off_sag_score(features: Dict) -> float:
    """
    计算停电/骤降的可能性分数（0-1）
    v13改进：合并停电和骤降，统一识别后段下降的情况
    """
    rms_first = features['rms_first']
    rms_last = features['rms_last']
    change_mid_first = features['change_mid_first']
    change_last_first = features['change_last_first']
    is_first_near_zero = features['is_first_near_zero']
    is_last_near_zero = features['is_last_near_zero']

    score = 0.0

    # 特征1：前段非近零（区别于合闸）
    if is_first_near_zero:
        return 0.0
    score += 0.3

    # 特征2：后段明显下降（关键特征，后/前 < 0.7）
    if change_last_first < 0.5:
        score += 1.0  # 直接满分
    elif change_last_first < 0.7:
        score += 0.3

    # 特征3：后段近零或明显下降
    if is_last_near_zero:
        score += 0.4

    # 特征4：中段无明显爆发（区别于跳闸）
    if change_mid_first < 1.5:
        score += 0.3

    return min(score, 1.0)


def calculate_closing_success_score(features: Dict) -> float:
    """
    计算合闸成功的可能性分数（0-1）
    v13改进：使用正弦比作为主要判据（>0.998）
    """
    rms_first = features['rms_first']
    rms_last = features['rms_last']
    change_last_first = features['change_last_first']
    change_last_mid = features['change_last_mid']
    sine_ratio_last = features['sine_ratio_last']
    is_first_near_zero = features['is_first_near_zero']
    is_last_near_zero = features['is_last_near_zero']

    score = 0.0

    # 特征1：前段近零（关键特征，区别于骤升/骤降）
    if not is_first_near_zero:
        return 0.0
    score += 0.2

    # 特征2：后段非近零
    if not is_last_near_zero:
        score += 0.1

    # 特征3：完美正弦波（关键特征，合闸成功的核心判据）
    if sine_ratio_last >= 0.999:
        score += 0.6
    elif sine_ratio_last >= 0.995:
        score += 0.4
    elif sine_ratio_last >= 0.99:
        score += 0.2

    # 特征4：后/中比值 >= 0.70（辅助特征）
    if change_last_mid >= 1.0:
        score += 0.2
    elif change_last_mid >= 0.70:
        score += 0.1

    return min(score, 1.0)


def calculate_closing_failure_score(features: Dict) -> float:
    """
    计算合闸失败的可能性分数（0-1）
    v22改进：基于"三段结构+后段衰减"特征
    - 前段近零：front_mean < 15
    - 中段爆发：middle_mean > 50 且 middle/front >= 2.0
    - 后段衰减：back/middle < 0.3（后段明显小于中段）
    """
    rms_first = features['rms_first']
    rms_last = features['rms_last']
    change_last_first = features['change_last_first']
    change_last_mid = features['change_last_mid']
    change_mid_first = features.get('change_mid_first', 1.0)
    sine_ratio_last = features['sine_ratio_last']
    is_first_near_zero = features['is_first_near_zero']
    is_last_near_zero = features['is_last_near_zero']
    num_segments = features.get('num_segments', 3)

    score = 0.0

    # 特征1：前段近零（关键特征）
    if not is_first_near_zero:
        return 0.0
    score += 0.3

    # 特征2：中段爆发（关键特征，区别于合闸成功/上电）
    if change_mid_first >= 2.0:
        score += 0.4
    elif change_mid_first >= 1.5:
        score += 0.2

    # 特征3：后段衰减（关键特征，区别于合闸成功/上电）
    # 合闸失败的后段明显小于中段（back/middle < 0.3）
    # 而合闸成功/上电的后段接近或大于前段
    if change_last_mid < 0.3:
        score += 1.0  # 直接满分，核心特征
    elif change_last_mid < 0.5:
        score += 0.7
    elif change_last_mid < 0.65:
        score += 0.4

    # 特征4：段数>=3（合闸失败通常是三段结构）
    if num_segments >= 3:
        score += 0.3
    else:
        score -= 0.5  # 段数不足，大概率不是合闸失败

    return max(0.0, min(score, 1.0))


def calculate_surge_score(features: Dict) -> float:
    """
    计算电流骤升的可能性分数（0-1）
    v21优化：区分真实骤升和中间异常的扰动
    - 真实骤升：后段持续高电流，整体趋势上升
    - 中间异常的扰动：后段恢复正常，前后段幅值相近
    """
    rms_first = features['rms_first']
    rms_last = features['rms_last']
    change_last_first = features['change_last_first']
    overall_trend = features['overall_trend']
    sine_ratio_last = features['sine_ratio_last']
    sine_ratio_first = features['sine_ratio_first']
    is_first_near_zero = features['is_first_near_zero']
    is_last_near_zero = features['is_last_near_zero']
    num_segments = features.get('num_segments', 3)

    score = 0.0

    # 特征1：前段非近零（关键特征，区别于合闸）
    if is_first_near_zero:
        return 0.0
    score += 0.3

    # 特征2：后段非近零（关键特征，区别于跳闸/停电）
    if is_last_near_zero and rms_last < 1.0:
        return 0.2
    score += 0.3

    # 特征3：关键判断 - 区分骤升和扰动
    # 真实骤升：change_last_first > 1.5 且 段数=2
    # 扰动：change_last_first <= 1.5 或 段数!=2
    if change_last_first > 1.5 and num_segments == 2:
        score += 0.4
    elif change_last_first > 2.0 and num_segments == 2:
        score += 0.6
    elif 1.0 < change_last_first <= 1.5 and num_segments == 2:
        score += 0.2  # 轻微上升，可能是扰动
    # 如果前后段比值接近（<=1.5），大幅降低骤升评分
    elif change_last_first <= 1.5:
        score -= 0.5  # 这是扰动的特征，不是骤升

    # 特征4：整体上升趋势
    if overall_trend == 'rising':
        score += 0.3
    elif overall_trend == 'stable' and change_last_first <= 1.5:
        score -= 0.3  # 整体稳定且前后相近，肯定是扰动

    # 特征5：前后段都是正弦波
    if sine_ratio_first >= 0.9 and sine_ratio_last >= 0.9:
        score += 0.2
    elif sine_ratio_last >= 0.9:
        score += 0.1

    return max(0.0, min(score, 1.0))


def calculate_power_on_score(features: Dict) -> float:
    """
    计算上电的可能性分数（0-1）
    v20.16新增
    """
    rms_first = features['rms_first']
    rms_last = features['rms_last']
    change_last_first = features['change_last_first']
    sine_ratio_last = features['sine_ratio_last']
    is_first_near_zero = features['is_first_near_zero']
    is_last_near_zero = features['is_last_near_zero']

    score = 0.0

    # 特征1：前段近零（关键特征）
    if is_first_near_zero:
        score += 0.4
    else:
        return 0.0

    # 特征2：后段非近零（关键特征）
    if not is_last_near_zero:
        score += 0.3
    else:
        return 0.0

    # 特征3：整体大幅上升（后/前 > 100）
    if change_last_first > 100:
        score += 0.3
    elif change_last_first > 10:
        score += 0.1

    # 特征4：后段是稳定的正弦波
    if sine_ratio_last >= 0.9:
        score += 0.2
    elif sine_ratio_last >= 0.6:
        score += 0.1

    return min(score, 1.0)



def classify_fault(features: Dict) -> Dict:
    """
    基于特征进行故障分类（v20 - 科学分段版）

    分层决策策略：
    1. 基于工作状态和段数进行初步判断
    2. 使用频域和时域特征进行精细识别

    工作状态定义：
    - power_on：上电（前段近零，后段稳定）
    - power_off：停电（前段稳定，后段近零）
    - normal：正常工作（前后段都稳定且相近）
    - unknown：未知状态

    故障类型（6种）：
    1. 扰动：段数=3 + 前后段相近 + 中段稳定
    2. 跳闸：段数>=3 + 后段近零 + 中段爆发
    3. 合闸成功：段数=2 + 前段近零 + 后段稳定 + 正弦比>0.998
    4. 合闸失败：段数>=3 + 前段近零 + 后段衰减
    5. 电流骤升：段数=2 + 前后段都非近零 + 后段上升
    6. 停电/骤降：段数=2 + 前段非近零 + 后段近零

    Args:
        features: 特征字典

    Returns:
        分类结果字典
    """
    # 计算各故障类型的评分
    scores = {
        'disturbance': calculate_disturbance_score(features),
        'trip': calculate_trip_score(features),
        'power_off_sag': calculate_power_off_sag_score(features),
        'closing_success': calculate_closing_success_score(features),
        'closing_failure': calculate_closing_failure_score(features),
        'current_surge': calculate_surge_score(features),
        'power_on': calculate_power_on_score(features),  # v20.16新增
    }

    # 提取关键特征
    work_state = features.get('work_state', 'unknown')
    num_segments = features.get('num_segments', 3)
    sine_ratio_last = features.get('sine_ratio_last', 0)
    change_last_mid = features.get('change_last_mid', 1.0)
    change_mid_first = features.get('change_mid_first', 1.0)
    change_last_first = features.get('change_last_first', 1.0)

    # v22决策树：优化合闸失败识别，使用"三段结构+后段衰减"特征
    
    # v23新增：周期级别波形分析扰动检测
    # 关键特征：开始/结束周期有非正弦波或不对称特征
    # 这个检测路径优先级最高，用于捕获传统RMS分析遗漏的扰动
    has_non_sinusoidal_cycles = features.get('has_non_sinusoidal_cycles', False)
    has_asymmetric_cycles = features.get('has_asymmetric_cycles', False)
    
    best_type = None
    
    # 如果有非正弦波或不对称特征，且不是明显的其他故障类型，则判为扰动
    # 关键条件：前后段RMS不能差异太大（排除跳闸和停电/骤降）
    if (has_non_sinusoidal_cycles or has_asymmetric_cycles):
        rms_ratio = features['rms_last'] / (features['rms_first'] + 1e-6)
        
        # 排除明显的跳闸（后段RMS < 5且前段RMS > 20）
        if features['rms_last'] < 5.0 and features['rms_first'] > 20.0:
            best_type = None
        # 排除明显的停电/骤降（后段RMS < 前段RMS的30%）
        elif rms_ratio < 0.3:
            best_type = None
        # 其他情况：判为扰动
        else:
            best_type = 'disturbance'
    
    # 0. 合闸失败识别（v22优化：前段近零 + 中段爆发 + 后段衰减 + 段数>=3）
    # 关键特征：前段近零 + 中段爆发（change_mid_first >= 2.0）+ 后段衰减（change_last_mid < 0.3）
    # 与上电的区别：上电的后/前 > 100，合闸失败的后/中 < 0.3
    # 优先判断，避免被误判为跳闸
    elif (num_segments >= 3 and
          features['is_first_near_zero'] and  # 前段近零
          change_mid_first >= 2.0 and  # 中段爆发
          change_last_mid < 0.3):  # 后段衰减（小于中段的30%）
        best_type = 'closing_failure'
    # 1. 跳闸识别（优化：放宽后段近零阈值，适应连续衰减）
    # 关键特征：前段非近零 + 后段RMS<30A + 整体大幅下降或中段爆发
    elif (num_segments >= 2 and
        not features['is_first_near_zero'] and  # 前段非近零（关键特征，区别于合闸失败）
        features['rms_last'] < 30.0 and  # 后段衰减到很小（放宽阈值）
        (change_mid_first >= 2.0 or change_last_first < 0.1)):  # 中段爆发或整体大幅下降
        best_type = 'trip'
    # 2. 扰动识别（v21优化：前后段幅值相近，允许中间短暂异常）
    # 关键特征：前后段工频电流幅值比<=1.5 + 工作状态normal/unknown + 非近零
    elif (work_state in ['normal', 'unknown'] and
        change_last_first <= 1.5 and  # 核心判据：前后段工频电流幅值比<=1.5（严格）
        not features['is_first_near_zero'] and
        not features['is_last_near_zero']):  # 前后段都非近零
        best_type = 'disturbance'
    # 3. 上电识别（新增：区分上电和合闸成功）
    # 关键特征：前段近零 + 后段非近零 + 整体大幅上升（后/前>100）
    elif (features['is_first_near_zero'] and
          not features['is_last_near_zero'] and
          change_last_first > 100.0):  # 上电：后/前>100
        best_type = 'power_on'
    # 4. 合闸成功识别（优化：限制后/前比值范围）
    # 关键特征：前段近零 + 后段稳定 + 中等幅度上升（2 < 后/前 <= 100）
    elif (features['is_first_near_zero'] and
          not features['is_last_near_zero'] and
          2.0 < change_last_first <= 100.0):  # 合闸成功：2 < 后/前 <= 100
        best_type = 'closing_success'
    # 5. 停电/骤降识别（强特征：段数=2 + 后段近零 + 前段非近零）
    elif (num_segments == 2 and
          features['is_last_near_zero'] and
          not features['is_first_near_zero'] and
          features['rms_first'] > 5):  # 前段有一定电流
        best_type = 'power_off_sag'
    # 6. 电流骤升识别（v21优化：更严格的后/前比值要求）
    # 关键特征：前后段都非近零 + 后段持续上升（后/前>1.5）
    # 与扰动的区别：骤升的后/前 > 1.5，扰动的后/前 <= 1.5
    elif (work_state in ['normal', 'unknown'] and
        change_last_first > 1.5 and  # 后/前 > 1.5，区别于扰动
        not features['is_first_near_zero'] and
        not features['is_last_near_zero'] and
        scores['current_surge'] >= 0.6):  # 评分阈值降低，主要靠决策树判断
        best_type = 'current_surge'
    # 7. 其他情况：使用最高评分
    else:
        # 排除扰动，只比较故障类型
        fault_scores = {k: v for k, v in scores.items() if k != 'disturbance'}
        best_type = max(fault_scores, key=fault_scores.get)

    best_score = scores[best_type]

    result = {
        'fault_type': best_type,
        'fault_type_cn': '',
        'confidence': best_score,
        'reason': [],
        'scores': scores,
        'features': features
    }

    # 中文名称
    type_cn_map = {
        'disturbance': '扰动',
        'trip': '跳闸',
        'power_off_sag': '停电/骤降',
        'closing_success': '合闸成功',
        'closing_failure': '合闸失败',
        'current_surge': '电流骤升',
        'power_on': '上电',  # 新增
    }
    result['fault_type_cn'] = type_cn_map[best_type]

    # 生成判断依据
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    result['reason'] = [f'{k}={v:.2f}' for k, v in sorted_scores[:3]]

    return result


def detect_fault(file_path: str) -> Dict:
    """
    完整的故障检测流程

    Args:
        file_path: 数据文件路径

    Returns:
        检测结果字典
    """
    signal = load_current_data(file_path)
    features = extract_features(signal, 3200)
    result = classify_fault(features)

    result['file_path'] = file_path
    result['signal_length'] = len(signal)

    return result


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='输电线路工频电流故障识别（v17 - 频域特征增强版）')
    parser.add_argument('file_path', help='数据文件路径')
    args = parser.parse_args()

    result = detect_fault(args.file_path)

    print(f"故障类型: {result['fault_type_cn']}")
    print(f"英文类型: {result['fault_type']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"\n各类型评分:")
    for fault_type, score in sorted(result['scores'].items(), key=lambda x: -x[1]):
        cn = {
            'disturbance': '扰动',
            'trip': '跳闸',
            'power_off_sag': '停电/骤降',
            'closing_success': '合闸成功',
            'closing_failure': '合闸失败',
            'current_surge': '电流骤升',
        }.get(fault_type, fault_type)
        print(f"  {cn:12s}: {score:.3f}")
