#include "power_freq_analyzer.h"
#include "signal_processing.h"
#include <algorithm>
#include <numeric>
#include <cmath>

static constexpr double EPS = 1e-6;

PowerFreqAnalyzer::PowerFreqAnalyzer() {
    // v22: 不再依赖ConfigManager的旧阈值配置
}

// ===== 基础信号处理 =====

double PowerFreqAnalyzer::calculateRMS(const std::vector<double>& samples) {
    if (samples.empty()) return 0.0;
    double sumSq = 0.0;
    for (double s : samples) sumSq += s * s;
    return std::sqrt(sumSq / samples.size());
}

double PowerFreqAnalyzer::computeMedian(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if (n % 2 == 0) {
        return (values[n / 2 - 1] + values[n / 2]) / 2.0;
    }
    return values[n / 2];
}

std::vector<double> PowerFreqAnalyzer::medianFilter(const std::vector<double>& data, int windowSize) {
    if (data.size() < static_cast<size_t>(windowSize)) return data;
    std::vector<double> result(data.size());
    int halfW = windowSize / 2;
    for (size_t i = 0; i < data.size(); ++i) {
        int start = std::max(0, static_cast<int>(i) - halfW);
        int end = std::min(static_cast<int>(data.size()), static_cast<int>(i) + halfW + 1);
        std::vector<double> window(data.begin() + start, data.begin() + end);
        result[i] = computeMedian(window);
    }
    return result;
}

int PowerFreqAnalyzer::findCycleStart(const std::vector<double>& signal, double samplingFreqHz, double targetFreq) {
    int cycleLength = static_cast<int>(samplingFreqHz / targetFreq);
    int searchLimit = std::min(static_cast<int>(signal.size()), cycleLength * 2);
    for (int i = 1; i < searchLimit; ++i) {
        if (signal[i - 1] <= 0.0 && signal[i] > 0.0) {
            return i;
        }
    }
    return 0;
}

std::vector<double> PowerFreqAnalyzer::computeRmsAligned(const std::vector<double>& signal, double samplingFreqHz,
                                                          double targetFreq, bool smoothRms) {
    int cycleLength = static_cast<int>(samplingFreqHz / targetFreq);
    if (cycleLength <= 0) cycleLength = 1;

    int startIdx = findCycleStart(signal, samplingFreqHz, targetFreq);

    std::vector<double> rmsValues;
    for (int i = startIdx; i + cycleLength <= static_cast<int>(signal.size()); i += cycleLength) {
        double sumSq = 0.0;
        for (int j = i; j < i + cycleLength; ++j) {
            sumSq += signal[j] * signal[j];
        }
        rmsValues.push_back(std::sqrt(sumSq / cycleLength));
    }

    // 如果数据长度不够，使用滑动窗口补充
    if (rmsValues.size() < 10) {
        rmsValues.clear();
        for (int i = 0; i + cycleLength <= static_cast<int>(signal.size()); ++i) {
            double sumSq = 0.0;
            for (int j = i; j < i + cycleLength; ++j) {
                sumSq += signal[j] * signal[j];
            }
            rmsValues.push_back(std::sqrt(sumSq / cycleLength));
        }
    }

    // 中值滤波平滑
    if (smoothRms && rmsValues.size() >= 5) {
        rmsValues = medianFilter(rmsValues, std::min(5, static_cast<int>(rmsValues.size())));
    }

    return rmsValues;
}

// ===== 科学分段算法 =====

PowerFreqAnalyzer::SegmentResult PowerFreqAnalyzer::detectSegments(const std::vector<double>& rmsValues,
                                                                     double changeThreshold, int minSegmentLen) {
    SegmentResult result;

    if (minSegmentLen < 0) {
        minSegmentLen = std::max(1, static_cast<int>(rmsValues.size()) / 10);
    }

    if (rmsValues.size() < static_cast<size_t>(minSegmentLen * 2)) {
        result.num_segments = 1;
        std::vector<double> tmp(rmsValues.begin(), rmsValues.end());
        result.segment_rms_values.push_back(computeMedian(tmp));
        result.segment_indices.push_back(static_cast<int>(rmsValues.size()));
        return result;
    }

    // 计算整体RMS水平
    double overallRms = 0.0;
    for (double v : rmsValues) overallRms += v;
    overallRms /= rmsValues.size();

    // 计算滑动窗口变化率
    int windowSize = std::max(1, static_cast<int>(rmsValues.size()) / 30);
    std::vector<double> changeRates;

    for (int i = windowSize; i < static_cast<int>(rmsValues.size()) - windowSize; ++i) {
        double prevMean = 0.0, currMean = 0.0;
        for (int j = i - windowSize; j < i; ++j) prevMean += rmsValues[j];
        prevMean /= windowSize;
        for (int j = i; j < i + windowSize; ++j) currMean += rmsValues[j];
        currMean /= windowSize;

        double absChange = std::abs(currMean - prevMean);
        double changeRate;
        if (prevMean > EPS && currMean > EPS) {
            changeRate = absChange / std::max(prevMean, currMean);
        } else {
            changeRate = absChange / (overallRms + EPS);
        }
        changeRates.push_back(changeRate);
    }

    // 找突变点
    std::vector<int> changeIndices;
    for (int i = 0; i < static_cast<int>(changeRates.size()); ++i) {
        if (changeRates[i] > changeThreshold) {
            changeIndices.push_back(i);
        }
    }

    if (changeIndices.empty()) {
        result.num_segments = 1;
        std::vector<double> tmp(rmsValues.begin(), rmsValues.end());
        result.segment_rms_values.push_back(computeMedian(tmp));
        result.segment_indices.push_back(static_cast<int>(rmsValues.size()));
        return result;
    }

    // 合并近距离突变点
    std::vector<int> mergedIndices;
    for (int idx : changeIndices) {
        if (mergedIndices.empty()) {
            mergedIndices.push_back(idx);
        } else {
            if (idx - mergedIndices.back() >= minSegmentLen) {
                mergedIndices.push_back(idx);
            } else if (changeRates[idx] > changeRates[mergedIndices.back()]) {
                mergedIndices.back() = idx;
            }
        }
    }

    // 边界过滤
    std::vector<int> segmentIndices;
    for (int idx : mergedIndices) {
        if (idx >= minSegmentLen && idx <= static_cast<int>(rmsValues.size()) - minSegmentLen) {
            segmentIndices.push_back(idx);
        }
    }

    if (segmentIndices.empty()) {
        result.num_segments = 1;
        std::vector<double> tmp(rmsValues.begin(), rmsValues.end());
        result.segment_rms_values.push_back(computeMedian(tmp));
        result.segment_indices.push_back(static_cast<int>(rmsValues.size()));
        return result;
    }

    // 计算各段RMS（使用中值）
    std::vector<double> segRmsValues;
    int prevIdx = 0;
    for (int idx : segmentIndices) {
        std::vector<double> seg(rmsValues.begin() + prevIdx, rmsValues.begin() + idx);
        segRmsValues.push_back(computeMedian(seg));
        prevIdx = idx;
    }
    // 最后一段
    {
        std::vector<double> seg(rmsValues.begin() + prevIdx, rmsValues.end());
        segRmsValues.push_back(computeMedian(seg));
    }

    // 合并相似段（0.6~1.67倍）
    std::vector<double> mergedSegments;
    mergedSegments.push_back(segRmsValues[0]);
    std::vector<int> mergedIndicesResult;
    if (!segmentIndices.empty()) {
        mergedIndicesResult.push_back(segmentIndices[0]);
    }

    for (size_t i = 1; i < segRmsValues.size(); ++i) {
        double prevRms = segRmsValues[i - 1];
        double currRms = segRmsValues[i];
        double ratio = currRms / (prevRms + EPS);
        if (ratio >= 0.6 && ratio <= 1.67) {
            mergedSegments.back() = (mergedSegments.back() + currRms) / 2.0;
        } else {
            mergedSegments.push_back(currRms);
            if (i - 1 < segmentIndices.size()) {
                mergedIndicesResult.push_back(segmentIndices[i - 1]);
            }
        }
    }

    // 构建最终分段索引
    std::vector<int> finalIndices;
    if (mergedSegments.size() > 1 && !mergedIndicesResult.empty()) {
        for (size_t i = 0; i + 1 < mergedIndicesResult.size(); ++i) {
            finalIndices.push_back(mergedIndicesResult[i]);
        }
    }
    finalIndices.push_back(static_cast<int>(rmsValues.size()));

    result.num_segments = static_cast<int>(mergedSegments.size());
    result.segment_rms_values = mergedSegments;
    result.segment_indices = finalIndices;
    return result;
}

// ===== 时域特征提取 =====

PowerFreqFeatures PowerFreqAnalyzer::extractTimeDomainFeatures(const std::vector<double>& signal, double samplingFreqHz) {
    PowerFreqFeatures features;

    // 计算RMS序列
    std::vector<double> rmsValues = computeRmsAligned(signal, samplingFreqHz, 50.0, true);

    if (rmsValues.size() < 10) {
        int cycleLength = static_cast<int>(samplingFreqHz / 50.0);
        if (cycleLength <= 0) cycleLength = 1;
        rmsValues.clear();
        for (int i = 0; i + cycleLength <= static_cast<int>(signal.size()); ++i) {
            double sumSq = 0.0;
            for (int j = i; j < i + cycleLength; ++j) {
                sumSq += signal[j] * signal[j];
            }
            rmsValues.push_back(std::sqrt(sumSq / cycleLength));
        }
    }

    // 科学分段
    SegmentResult seg = detectSegments(rmsValues);
    features.num_segments = seg.num_segments;
    features.segment_rms_values = seg.segment_rms_values;
    features.segment_indices = seg.segment_indices;

    // 前段RMS
    if (seg.num_segments >= 1) {
        features.rms_first = seg.segment_rms_values[0];
    } else {
        std::vector<double> tmp(rmsValues.begin(), rmsValues.end());
        features.rms_first = computeMedian(tmp);
    }

    // 后段RMS
    if (seg.num_segments >= 2) {
        features.rms_last = seg.segment_rms_values.back();
    } else {
        features.rms_last = features.rms_first;
    }

    // 中段RMS
    if (seg.num_segments >= 3) {
        std::vector<double> midValues(seg.segment_rms_values.begin() + 1,
                                       seg.segment_rms_values.end() - 1);
        features.rms_middle = computeMedian(midValues);
    } else {
        features.rms_middle = (features.rms_first + features.rms_last) / 2.0;
    }

    // 相对变化率
    features.change_last_first = features.rms_last / (features.rms_first + EPS);
    features.change_mid_first = features.rms_middle / (features.rms_first + EPS);
    features.change_last_mid = features.rms_last / (features.rms_middle + EPS);

    // 整体趋势
    if (features.change_last_first > 1.2) {
        features.overall_trend = "rising";
    } else if (features.change_last_first < 0.8) {
        features.overall_trend = "falling";
    } else {
        features.overall_trend = "stable";
    }

    // 近零判断
    const double nearZeroThreshold = 5.0;
    features.is_first_near_zero = features.rms_first < nearZeroThreshold;
    features.is_middle_near_zero = features.rms_middle < nearZeroThreshold;
    features.is_last_near_zero = features.rms_last < nearZeroThreshold;

    // 工作状态
    if (features.is_first_near_zero && !features.is_last_near_zero) {
        features.work_state = WorkState::POWER_ON;
    } else if (!features.is_first_near_zero && features.is_last_near_zero) {
        features.work_state = WorkState::POWER_OFF;
    } else if (features.change_last_first >= 0.5 && features.change_last_first <= 2.0) {
        features.work_state = WorkState::NORMAL;
    } else {
        features.work_state = WorkState::STATE_UNKNOWN;
    }

    // 正弦比（std/rms）
    int cycleLength = static_cast<int>(samplingFreqHz / 50.0);
    if (cycleLength <= 0) cycleLength = 1;

    // 前段std
    int firstEnd = 0;
    if (!seg.segment_indices.empty()) {
        firstEnd = seg.segment_indices[0] * cycleLength;
    } else {
        firstEnd = static_cast<int>(signal.size());
    }
    firstEnd = std::min(firstEnd, static_cast<int>(signal.size()));
    if (firstEnd > 0) {
        features.sine_ratio_first = SignalProcessing::calculateStdDev(
            std::vector<double>(signal.begin(), signal.begin() + firstEnd)) / (features.rms_first + EPS);
    }

    // 后段std
    int lastStart = 0;
    if (!seg.segment_indices.empty()) {
        lastStart = seg.segment_indices.back() * cycleLength;
    }
    lastStart = std::min(lastStart, static_cast<int>(signal.size()));
    if (lastStart < static_cast<int>(signal.size())) {
        features.sine_ratio_last = SignalProcessing::calculateStdDev(
            std::vector<double>(signal.begin() + lastStart, signal.end())) / (features.rms_last + EPS);
    } else if (static_cast<int>(signal.size()) >= cycleLength) {
        features.sine_ratio_last = SignalProcessing::calculateStdDev(
            std::vector<double>(signal.end() - cycleLength, signal.end())) / (features.rms_last + EPS);
    }

    // 中段std
    if (seg.num_segments >= 3) {
        int midStart = firstEnd;
        int midEnd = lastStart;
        if (midEnd > midStart) {
            features.sine_ratio_middle = SignalProcessing::calculateStdDev(
                std::vector<double>(signal.begin() + midStart, signal.begin() + midEnd)) / (features.rms_middle + EPS);
        } else {
            features.sine_ratio_middle = (features.sine_ratio_first + features.sine_ratio_last) / 2.0;
        }
    } else {
        features.sine_ratio_middle = (features.sine_ratio_first + features.sine_ratio_last) / 2.0;
    }

    return features;
}

// ===== 频域特征提取 =====

void PowerFreqAnalyzer::extractFrequencyFeatures(const std::vector<double>& signal, double samplingFreqHz,
                                                   PowerFreqFeatures& features) {
    size_t n = signal.size();
    if (n == 0) return;

    // FFT变换
    std::vector<double> spectrum = SignalProcessing::performFFT(signal);
    size_t numFreqs = spectrum.size();

    // 计算功率谱和总功率
    std::vector<double> powerSpectrum(numFreqs);
    double totalPower = 0.0;
    for (size_t i = 0; i < numFreqs; ++i) {
        powerSpectrum[i] = spectrum[i] * spectrum[i];
        totalPower += powerSpectrum[i];
    }

    if (totalPower < EPS) return;

    // 频率分辨率
    double freqResolution = samplingFreqHz / n;

    // 1. 50Hz能量占比
    double power50Hz = 0.0;
    for (size_t i = 0; i < numFreqs; ++i) {
        double freq = i * freqResolution;
        if (freq >= 45.0 && freq <= 55.0) {
            power50Hz += powerSpectrum[i];
        }
    }
    features.power_ratio_50Hz = power50Hz / totalPower;

    // 2. 频谱集中度（top-10能量占比）
    std::vector<double> sortedPower = powerSpectrum;
    std::sort(sortedPower.begin(), sortedPower.end(), std::greater<double>());
    double topPower = 0.0;
    int topK = std::min(10, static_cast<int>(sortedPower.size()));
    for (int i = 0; i < topK; ++i) topPower += sortedPower[i];
    features.power_concentration = topPower / totalPower;

    // 3. 主频能量比
    double maxPower = *std::max_element(powerSpectrum.begin(), powerSpectrum.end());
    features.power_ratio_max = maxPower / totalPower;

    // 4. 主频频率
    auto maxIt = std::max_element(powerSpectrum.begin(), powerSpectrum.end());
    size_t maxIdx = std::distance(powerSpectrum.begin(), maxIt);
    features.dominant_freq = maxIdx * freqResolution;

    // 5. 频谱熵
    double entropy = 0.0;
    for (size_t i = 0; i < numFreqs; ++i) {
        double p = powerSpectrum[i] / totalPower;
        if (p > 1e-10) {
            entropy -= p * std::log2(p);
        }
    }
    features.spectral_entropy = entropy;
}

// ===== 完整特征提取 =====

PowerFreqFeatures PowerFreqAnalyzer::extractFeatures(const std::vector<double>& signal, double samplingFreqHz) {
    PowerFreqFeatures features = extractTimeDomainFeatures(signal, samplingFreqHz);
    extractFrequencyFeatures(signal, samplingFreqHz, features);
    return features;
}

// ===== 评分函数 =====

double PowerFreqAnalyzer::calculateDisturbanceScore(const PowerFreqFeatures& f) {
    double score = 0.0;

    // 特征1：前后段相近
    if (f.change_last_first >= 0.8 && f.change_last_first <= 1.2) {
        score += 0.5;
    } else if (f.change_last_first >= 0.7 && f.change_last_first <= 1.3) {
        score += 0.4;
    } else if (f.change_last_first <= 1.5) {
        score += 0.3;
    } else if (f.change_last_first > 1.5 && f.change_last_first <= 2.0) {
        score += 0.1;
    } else {
        score -= 0.5;
    }

    // 特征2：中段稳定性
    if (f.change_mid_first >= 0.7 && f.change_mid_first <= 1.3 &&
        f.change_last_mid >= 0.7 && f.change_last_mid <= 1.3) {
        score += 0.4;
    } else if (f.change_mid_first >= 0.5 && f.change_mid_first <= 2.0 &&
               f.change_last_mid >= 0.5 && f.change_last_mid <= 2.0) {
        score += 0.3;
    } else if (f.change_mid_first > 3.0 || f.change_last_mid > 3.0) {
        score -= 1.0;
    } else if (f.change_mid_first > 2.0 || f.change_last_mid > 2.0) {
        score -= 0.2;
    }

    // 特征3：非近零
    if (!f.is_first_near_zero && !f.is_last_near_zero) {
        score += 0.2;
    } else {
        score -= 0.3;
    }

    // 特征4：频谱集中度
    if (f.power_concentration > 0.9) {
        score += 0.1;
    } else if (f.power_concentration > 0.7) {
        score += 0.05;
    }

    // 特征5：频谱熵
    if (f.spectral_entropy < 3.5) {
        score += 0.1;
    } else if (f.spectral_entropy < 4.5) {
        score += 0.05;
    }

    return std::max(0.0, std::min(score, 1.0));
}

double PowerFreqAnalyzer::calculateTripScore(const PowerFreqFeatures& f) {
    double score = 0.0;

    // 特征1：前段非近零
    if (!f.is_first_near_zero && f.rms_first > 5.0) {
        score += 0.3;
    }

    // 特征2：中段爆发
    if (f.change_mid_first > 2.0) {
        score += 0.5;
    } else if (f.change_mid_first > 1.5) {
        score += 0.3;
    }

    // 特征3：后段近零或明显下降
    if (f.is_last_near_zero) {
        score += 0.6;
    } else if (f.rms_last < 5.0) {
        score += 0.4;
    }

    // 特征4：后段衰减
    if (f.change_last_mid < 0.5) {
        score += 0.2;
    }

    return std::min(score, 1.0);
}

double PowerFreqAnalyzer::calculatePowerOffSagScore(const PowerFreqFeatures& f) {
    double score = 0.0;

    // 特征1：前段非近零
    if (f.is_first_near_zero) return 0.0;
    score += 0.3;

    // 特征2：后段明显下降
    if (f.change_last_first < 0.5) {
        score += 1.0;
    } else if (f.change_last_first < 0.7) {
        score += 0.3;
    }

    // 特征3：后段近零
    if (f.is_last_near_zero) {
        score += 0.4;
    }

    // 特征4：中段无明显爆发
    if (f.change_mid_first < 1.5) {
        score += 0.3;
    }

    return std::min(score, 1.0);
}

double PowerFreqAnalyzer::calculateClosingSuccessScore(const PowerFreqFeatures& f) {
    double score = 0.0;

    // 特征1：前段近零
    if (!f.is_first_near_zero) return 0.0;
    score += 0.2;

    // 特征2：后段非近零
    if (!f.is_last_near_zero) score += 0.1;

    // 特征3：完美正弦波
    if (f.sine_ratio_last >= 0.999) {
        score += 0.6;
    } else if (f.sine_ratio_last >= 0.995) {
        score += 0.4;
    } else if (f.sine_ratio_last >= 0.99) {
        score += 0.2;
    }

    // 特征4：后/中比值
    if (f.change_last_mid >= 1.0) {
        score += 0.2;
    } else if (f.change_last_mid >= 0.70) {
        score += 0.1;
    }

    return std::min(score, 1.0);
}

double PowerFreqAnalyzer::calculateClosingFailureScore(const PowerFreqFeatures& f) {
    double score = 0.0;

    // 特征1：前段近零
    if (!f.is_first_near_zero) return 0.0;
    score += 0.3;

    // 特征2：中段爆发
    if (f.change_mid_first >= 2.0) {
        score += 0.4;
    } else if (f.change_mid_first >= 1.5) {
        score += 0.2;
    }

    // 特征3：后段衰减（核心特征）
    if (f.change_last_mid < 0.3) {
        score += 1.0;
    } else if (f.change_last_mid < 0.5) {
        score += 0.7;
    } else if (f.change_last_mid < 0.65) {
        score += 0.4;
    }

    // 特征4：段数>=3
    if (f.num_segments >= 3) {
        score += 0.3;
    } else {
        score -= 0.5;
    }

    return std::max(0.0, std::min(score, 1.0));
}

double PowerFreqAnalyzer::calculateSurgeScore(const PowerFreqFeatures& f) {
    double score = 0.0;

    // 特征1：前段非近零
    if (f.is_first_near_zero) return 0.0;
    score += 0.3;

    // 特征2：后段非近零
    if (f.is_last_near_zero && f.rms_last < 1.0) return 0.2;
    score += 0.3;

    // 特征3：关键判断 - 区分骤升和扰动
    if (f.change_last_first > 2.0 && f.num_segments == 2) {
        score += 0.6;
    } else if (f.change_last_first > 1.5 && f.num_segments == 2) {
        score += 0.4;
    } else if (f.change_last_first > 1.0 && f.change_last_first <= 1.5 && f.num_segments == 2) {
        score += 0.2;
    } else if (f.change_last_first <= 1.5) {
        score -= 0.5;
    }

    // 特征4：整体上升趋势
    if (f.overall_trend == "rising") {
        score += 0.3;
    } else if (f.overall_trend == "stable" && f.change_last_first <= 1.5) {
        score -= 0.3;
    }

    // 特征5：前后段都是正弦波
    if (f.sine_ratio_first >= 0.9 && f.sine_ratio_last >= 0.9) {
        score += 0.2;
    } else if (f.sine_ratio_last >= 0.9) {
        score += 0.1;
    }

    return std::max(0.0, std::min(score, 1.0));
}

double PowerFreqAnalyzer::calculatePowerOnScore(const PowerFreqFeatures& f) {
    double score = 0.0;

    // 特征1：前段近零
    if (f.is_first_near_zero) {
        score += 0.4;
    } else {
        return 0.0;
    }

    // 特征2：后段非近零
    if (!f.is_last_near_zero) {
        score += 0.3;
    } else {
        return 0.0;
    }

    // 特征3：整体大幅上升
    if (f.change_last_first > 100.0) {
        score += 0.3;
    } else if (f.change_last_first > 10.0) {
        score += 0.1;
    }

    // 特征4：后段是稳定正弦波
    if (f.sine_ratio_last >= 0.9) {
        score += 0.2;
    } else if (f.sine_ratio_last >= 0.6) {
        score += 0.1;
    }

    return std::min(score, 1.0);
}

// ===== v22 决策树分类 =====

ClassifyResult PowerFreqAnalyzer::classifyFault(const PowerFreqFeatures& features) {
    ClassifyResult result;

    // 计算各类型评分
    double scores[8];
    scores[0] = calculateTripScore(features);              // TRIP
    scores[1] = calculateClosingSuccessScore(features);    // CLOSING_SUCCESS
    scores[2] = calculateClosingFailureScore(features);    // CLOSING_FAILURE
    scores[3] = calculatePowerOnScore(features);           // POWER_ON
    scores[4] = calculateSurgeScore(features);             // CURRENT_SURGE
    scores[5] = calculateDisturbanceScore(features);       // DISTURBANCE
    scores[6] = calculatePowerOffSagScore(features);       // POWER_OFF_SAG
    scores[7] = 0.0;                                       // UNKNOWN

    for (int i = 0; i < 8; ++i) result.scores[i] = scores[i];

    TripEventType bestType = TripEventType::UNKNOWN;

    // 提取决策树需要的关键特征
    int numSeg = features.num_segments;
    double changeMidFirst = features.change_mid_first;
    double changeLastMid = features.change_last_mid;
    double changeLastFirst = features.change_last_first;
    bool isWorkNormal = (features.work_state == WorkState::NORMAL || features.work_state == WorkState::STATE_UNKNOWN);

    // [新增优化] -1. 非正弦/非直线强判定（专门处理 others 目录中非正弦畸变伪装成电流骤升）
    // 为了防止误杀真实的跳闸/合闸失败，只针对如下情况进行强判扰动：
    // 1. 没有强烈的故障特征（即 changeLastMid >= 0.5 且 changeMidFirst < 3.0）
    // 2. 前段非近零且 sine_ratio < 0.6，或者后段非近零且 sine_ratio < 0.6
    bool couldBeTripOrFail = (changeLastMid < 0.5 || features.is_last_near_zero || changeMidFirst >= 3.0);
    
    if (!couldBeTripOrFail && 
        ((!features.is_first_near_zero && features.sine_ratio_first < 0.6) ||
         (!features.is_last_near_zero && features.sine_ratio_last < 0.6))) {
        bestType = TripEventType::DISTURBANCE;
    }
    // 0. 合闸失败（v22优化）
    else if (numSeg >= 3 &&
        features.is_first_near_zero &&
        changeMidFirst >= 2.0 &&
        changeLastMid < 0.3) {
        bestType = TripEventType::CLOSING_FAILURE;
    }
    // 1. 跳闸
    else if (numSeg >= 2 &&
             !features.is_first_near_zero &&
             features.rms_last < 30.0 &&
             (features.is_last_near_zero || changeLastFirst < 0.5) &&
             (changeMidFirst >= 2.0 || changeLastFirst < 0.1)) {
        bestType = TripEventType::TRIP;
    }
    // 2. 扰动
    else if (isWorkNormal &&
             changeLastFirst <= 1.5 &&
             !features.is_first_near_zero &&
             !features.is_last_near_zero) {
        bestType = TripEventType::DISTURBANCE;
    }
    // 3. 上电
    else if (features.is_first_near_zero &&
             !features.is_last_near_zero &&
             changeLastFirst > 100.0) {
        bestType = TripEventType::POWER_ON;
    }
    // 4. 合闸成功
    else if (features.is_first_near_zero &&
             !features.is_last_near_zero &&
             changeLastFirst > 2.0 && changeLastFirst <= 100.0) {
        bestType = TripEventType::CLOSING_SUCCESS;
    }
    // 5. 停电/骤降
    else if (numSeg == 2 &&
             features.is_last_near_zero &&
             !features.is_first_near_zero &&
             features.rms_first > 5.0) {
        bestType = TripEventType::POWER_OFF_SAG;
    }
    // 6. 电流骤升
    else if (isWorkNormal &&
             changeLastFirst > 1.5 &&
             !features.is_first_near_zero &&
             !features.is_last_near_zero &&
             scores[4] >= 0.6) {  // current_surge score
        bestType = TripEventType::CURRENT_SURGE;
    }
    // 7. 兜底：使用最高评分（排除扰动，只比较故障类型）
    else {
        double maxScore = -1.0;
        // 排除扰动(5)和未知(7)
        int faultIndices[] = {0, 1, 2, 3, 4, 6};  // TRIP, CLOSING_SUCCESS/FAILURE, POWER_ON, SURGE, POWER_OFF_SAG
        TripEventType faultTypes[] = {
            TripEventType::TRIP, TripEventType::CLOSING_SUCCESS, TripEventType::CLOSING_FAILURE,
            TripEventType::POWER_ON, TripEventType::CURRENT_SURGE, TripEventType::POWER_OFF_SAG
        };
        for (int i = 0; i < 6; ++i) {
            if (scores[faultIndices[i]] > maxScore) {
                maxScore = scores[faultIndices[i]];
                bestType = faultTypes[i];
            }
        }
    }

    result.fault_type = bestType;
    result.confidence = scores[static_cast<int>(bestType)];

    return result;
}

// ===== 主入口函数 =====

std::vector<double> PowerFreqAnalyzer::convertToActualValues(const PowerFreqCurrentData& data) {
    std::vector<double> result;
    result.reserve(data.samples.size());
    for (short rawValue : data.samples) {
        result.push_back(data.convertToActualCurrent(rawValue));
    }
    return result;
}

TripEventType PowerFreqAnalyzer::analyzeTripEvent(const PowerFreqCurrentData& data) {
    std::vector<double> signal = convertToActualValues(data);
    if (signal.empty()) return TripEventType::UNKNOWN;
    return analyzeSignal(signal, data.samplingFreqHz);
}

TripEventType PowerFreqAnalyzer::analyzeSignal(const std::vector<double>& signal, double samplingFreqHz) {
    if (signal.empty()) return TripEventType::UNKNOWN;
    PowerFreqFeatures features = extractFeatures(signal, samplingFreqHz);
    ClassifyResult result = classifyFault(features);
    return result.fault_type;
}

const char* PowerFreqAnalyzer::getEventTypeName(TripEventType type) {
    switch (type) {
        case TripEventType::TRIP:             return "跳闸";
        case TripEventType::CLOSING_SUCCESS:  return "合闸成功";
        case TripEventType::CLOSING_FAILURE:  return "合闸失败";
        case TripEventType::POWER_ON:         return "上电";
        case TripEventType::CURRENT_SURGE:    return "电流骤升";
        case TripEventType::DISTURBANCE:      return "扰动";
        case TripEventType::POWER_OFF_SAG:    return "停电/骤降";
        default:                              return "未知";
    }
}
