#ifndef SIGNAL_PROCESSING_H
#define SIGNAL_PROCESSING_H

#include <vector>
#include <cmath>

// 信号处理工具类
class SignalProcessing {
public:
    // 小波变换（Daubechies 4小波）
    static std::vector<double> waveletTransform(const std::vector<double>& signal);
    
    // 检测突变点（返回突变点索引）
    static int detectTransition(const std::vector<double>& signal);
    
    // 低通滤波器
    static std::vector<double> lowPassFilter(const std::vector<double>& signal, 
                                             double cutoffFreq, 
                                             double samplingFreq);
    
    // 高通滤波器
    static std::vector<double> highPassFilter(const std::vector<double>& signal,
                                              double cutoffFreq,
                                              double samplingFreq);
    
    // 计算均值
    static double calculateMean(const std::vector<double>& data);
    
    // 计算方差
    static double calculateVariance(const std::vector<double>& data);
    
    // 计算标准差
    static double calculateStdDev(const std::vector<double>& data);
    
    // 寻找峰值索引
    static int findPeakIndex(const std::vector<double>& data);
    
    // 从峰值向前查找5%峰值处的索引（用于精确定位波头）
    static int find5PercentPeakIndex(const std::vector<double>& data);

    // 标准快速傅里叶变换 (FFT)
    // 返回复数模长（频率幅值）
    // 输入大小不必是2的幂，内部会处理
    static std::vector<double> performFFT(const std::vector<double>& signal);

    // 计算小波能量熵 (反映信号能量在不同频段的分布复杂性)
    // 雷击通常能量集中，熵值较低；噪声或复杂故障熵值较高
    static double calculateWaveletEnergyEntropy(const std::vector<double>& signal, int levels = 4);

    // 计算极性比率 (Polarity Ratio)
    // 返回 (MaxPositive / MaxAbsolute) 的绝对偏差
    // 接近 1.0 (或 0.0) 表示单极性强，接近 0.5 表示双极性震荡
    static double calculatePolarityFeature(const std::vector<double>& signal);
    
private:
    // Daubechies 4小波系数
    static constexpr double DB4_H0 = 0.6830127;
    static constexpr double DB4_H1 = 1.1830127;
    static constexpr double DB4_H2 = 0.3169873;
    static constexpr double DB4_H3 = -0.1830127;
};

#endif // SIGNAL_PROCESSING_H
