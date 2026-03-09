#include "signal_processing.h"
#include <algorithm>
#include <numeric>
#include <cmath>

std::vector<double> SignalProcessing::waveletTransform(const std::vector<double>& signal) {
    if (signal.size() < 4) {
        return signal;
    }
    
    std::vector<double> result(signal.size(), 0.0);
    
    // 简化的小波变换实现（单层分解）
    // 使用差分近似小波变换，并同时取绝对值
    result[0] = 0.0; // 第一个点没有前一个点，设为0
    for (size_t i = 1; i < signal.size(); ++i) {
        result[i] = std::abs(signal[i] - signal[i-1]);
    }
    
    return result;
}

int SignalProcessing::detectTransition(const std::vector<double>& signal) {
    if (signal.size() < 2) {
        return -1;
    }
    
    // 使用小波变换检测突变点
    std::vector<double> wavelet = waveletTransform(signal);
    
    // 寻找小波系数的最大值位置
    return findPeakIndex(wavelet);
}

std::vector<double> SignalProcessing::lowPassFilter(const std::vector<double>& signal,
                                                     double cutoffFreq,
                                                     double samplingFreq) {
    // 简单的移动平均滤波器
    if (signal.empty()) {
        return signal;
    }
    
    // 计算窗口大小
    int windowSize = static_cast<int>(samplingFreq / cutoffFreq);
    if (windowSize < 1) windowSize = 1;
    if (windowSize > static_cast<int>(signal.size())) {
        windowSize = signal.size();
    }
    
    std::vector<double> filtered(signal.size(), 0.0);
    
    for (size_t i = 0; i < signal.size(); ++i) {
        double sum = 0.0;
        int count = 0;
        
        int start = std::max(0, static_cast<int>(i) - windowSize / 2);
        int end = std::min(static_cast<int>(signal.size()), 
                          static_cast<int>(i) + windowSize / 2 + 1);
        
        for (int j = start; j < end; ++j) {
            sum += signal[j];
            count++;
        }
        
        // 添加除零检查(防御性编程)
        if (count > 0) {
            filtered[i] = sum / count;
        } else {
            filtered[i] = 0.0;
        }
    }
    
    return filtered;
}

std::vector<double> SignalProcessing::highPassFilter(const std::vector<double>& signal,
                                                      double cutoffFreq,
                                                      double samplingFreq) {
    // 高通滤波 = 原信号 - 低通滤波
    std::vector<double> lowPass = lowPassFilter(signal, cutoffFreq, samplingFreq);
    std::vector<double> highPass(signal.size());
    
    for (size_t i = 0; i < signal.size(); ++i) {
        highPass[i] = signal[i] - lowPass[i];
    }
    
    return highPass;
}

double SignalProcessing::calculateMean(const std::vector<double>& data) {
    if (data.empty()) {
        return 0.0;
    }
    
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

double SignalProcessing::calculateVariance(const std::vector<double>& data) {
    if (data.empty()) {
        return 0.0;
    }
    
    double mean = calculateMean(data);
    double variance = 0.0;
    
    for (double value : data) {
        double diff = value - mean;
        variance += diff * diff;
    }
    
    return variance / data.size();
}

double SignalProcessing::calculateStdDev(const std::vector<double>& data) {
    return std::sqrt(calculateVariance(data));
}

int SignalProcessing::findPeakIndex(const std::vector<double>& data) {
    if (data.empty()) {
        return -1;
    }
    
    auto maxIt = std::max_element(data.begin(), data.end());
    return std::distance(data.begin(), maxIt);
}

int SignalProcessing::find5PercentPeakIndex(const std::vector<double>& data) {
    if (data.empty()) {
        return -1;
    }
    
    // 首先找到峰值位置
    int peakIndex = findPeakIndex(data);
    if (peakIndex < 0) {
        return -1;
    }
    
    double peakValue = std::abs(data[peakIndex]);
    
    // 如果峰值太小，直接返回峰值索引
    if (peakValue < 1e-6) {
        return peakIndex;
    }
    
    // 计算5%峰值阈值
    double threshold = peakValue * 0.05;
    
    // 从峰值向前查找第一个小于5%峰值的点
    for (int i = peakIndex; i >= 0; --i) {
        if (std::abs(data[i]) < threshold) {
            // 返回刚好超过5%阈值的点（即i+1）
            if (i + 1 <= peakIndex) {
                return i + 1;
            }
            return peakIndex;
        }
    }
    
    // 如果所有点都大于5%峰值，返回第一个点
    return 0;
}

std::vector<double> SignalProcessing::performFFT(const std::vector<double>& signal) {
    if (signal.empty()) {
        return {};
    }

    size_t N = signal.size();
    size_t numFreqs = N / 2 + 1;
    std::vector<double> spectrum(numFreqs, 0.0);

    // 使用标量 DFT 实现（简单但正确）
    // 注意: 生产环境建议使用 FFTW 等专业库以获得更好的性能
    for (size_t k = 0; k < numFreqs; ++k) {
        double real = 0.0;
        double imag = 0.0;
        
        for (size_t n = 0; n < N; ++n) {
            double angle = 2.0 * M_PI * k * n / N;
            real += signal[n] * std::cos(angle);
            imag += signal[n] * std::sin(angle);
        }
        
        // 归一化
        double magnitude = std::sqrt(real * real + imag * imag);
        if (k == 0) {
            spectrum[k] = magnitude / N; // 直流分量
        } else {
            spectrum[k] = magnitude * 2.0 / N; // 交流分量
        }
    }

    return spectrum;
}

double SignalProcessing::calculateWaveletEnergyEntropy(const std::vector<double>& signal, int levels) {
    if (signal.empty()) return 0.0;
    
    std::vector<double> currentApprox = signal;
    std::vector<double> energies;
    energies.reserve(levels + 1); // 预分配足够的空间
    double totalEnergy = 0.0;
    
    // 简单的 Haar 小波分解计算各层能量
    for (int l = 0; l < levels; ++l) {
        if (currentApprox.size() < 2) break;
        
        size_t nextSize = currentApprox.size() / 2;
        std::vector<double> nextApprox;
        nextApprox.reserve(nextSize); // 预分配空间
        double levelEnergy = 0.0;
        
        for (size_t i = 0; i < nextSize; ++i) {
            size_t idx = 2 * i;
            double a = currentApprox[idx];
            double b = currentApprox[idx + 1];
            
            // Haar Detail (High Freq): (a - b) / sqrt(2)
            double detail = (a - b) / 1.41421356;
            
            // Haar Approx (Low Freq): (a + b) / sqrt(2)
            double approx = (a + b) / 1.41421356;
            
            levelEnergy += detail * detail;
            nextApprox.push_back(approx);
        }
        
        energies.push_back(levelEnergy);
        totalEnergy += levelEnergy;
        
        // 使用移动语义，避免深拷贝
        currentApprox = std::move(nextApprox);
    }
    
    // 加上剩余近似部分的能量
    double residualEnergy = 0.0;
    for (double val : currentApprox) {
        residualEnergy += val * val;
    }
    energies.push_back(residualEnergy);
    totalEnergy += residualEnergy;
    
    if (totalEnergy < 1e-9) return 0.0;
    
    // 计算香农熵
    double entropy = 0.0;
    for (double e : energies) {
        double p = e / totalEnergy;
        if (p > 1e-9) {
            entropy -= p * std::log(p);
        }
    }
    
    return entropy;
}

double SignalProcessing::calculatePolarityFeature(const std::vector<double>& signal) {
    if (signal.empty()) return 0.0;
    
    double maxPos = 0.0;
    double minNeg = 0.0; //Store as negative value
    
    for (double val : signal) {
        if (val > maxPos) maxPos = val;
        if (val < minNeg) minNeg = val;
    }
    
    double absMaxPos = std::abs(maxPos);
    double absMinNeg = std::abs(minNeg);
    double maxAbs = std::max(absMaxPos, absMinNeg);
    
    if (maxAbs < 1e-9) return 0.0;
    
    // 极性比率：正负峰值幅值之比 (0 ~ 1)
    // 1.0 表示完全对称的双极性
    // 0.0 (或接近) 表示单极性显著
    double ratio = std::min(absMaxPos, absMinNeg) / maxAbs;
    
    // 我们返回 "单极性程度"，即 1 - ratio
    // 1.0 表示强单极性，0.0 表示完全双极性
    return 1.0 - ratio;
}
