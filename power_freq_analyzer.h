#ifndef POWER_FREQ_ANALYZER_H
#define POWER_FREQ_ANALYZER_H

#include "waveform_data.h"
#include "config_manager.h"
#include <vector>
#include <cmath>
#include <string>

// 跳闸事件类型（v22 - 与Python技能对齐的7种故障类型）
enum class TripEventType {
    TRIP,               // 跳闸
    CLOSING_SUCCESS,    // 合闸成功
    CLOSING_FAILURE,    // 合闸失败
    POWER_ON,           // 上电
    CURRENT_SURGE,      // 电流骤升
    DISTURBANCE,        // 扰动
    POWER_OFF_SAG,      // 停电/骤降
    UNKNOWN             // 未知
};

// 工作状态
enum class WorkState {
    POWER_ON,    // 上电
    POWER_OFF,   // 停电
    NORMAL,      // 正常
    STATE_UNKNOWN // 未知
};

// 工频电流特征（时域 + 频域）
struct PowerFreqFeatures {
    // 三段RMS值
    double rms_first = 0.0;
    double rms_middle = 0.0;
    double rms_last = 0.0;

    // 相对变化率
    double change_last_first = 1.0;
    double change_mid_first = 1.0;
    double change_last_mid = 1.0;

    // 整体趋势
    std::string overall_trend = "stable";  // rising/falling/stable

    // 近零判断
    bool is_first_near_zero = false;
    bool is_middle_near_zero = false;
    bool is_last_near_zero = false;

    // 正弦比（std/rms）
    double sine_ratio_first = 0.0;
    double sine_ratio_middle = 0.0;
    double sine_ratio_last = 0.0;

    // 分段信息
    int num_segments = 1;
    std::vector<double> segment_rms_values;
    std::vector<int> segment_indices;

    // 工作状态
    WorkState work_state = WorkState::STATE_UNKNOWN;

    // 频域特征
    double power_ratio_50Hz = 0.0;
    double power_concentration = 0.0;
    double power_ratio_max = 0.0;
    double dominant_freq = 0.0;
    double spectral_entropy = 0.0;
};

// 分类结果
struct ClassifyResult {
    TripEventType fault_type = TripEventType::UNKNOWN;
    double confidence = 0.0;
    double scores[8] = {};  // 对应各故障类型的评分
};

// 工频电流分析器（v22 - 完整故障识别）
class PowerFreqAnalyzer {
public:
    PowerFreqAnalyzer();

    // 主分析函数：分析跳闸事件类型
    TripEventType analyzeTripEvent(const PowerFreqCurrentData& data);

    // 从double信号数组分析（直接输入实际电流值）
    TripEventType analyzeSignal(const std::vector<double>& signal, double samplingFreqHz);

    // 提取全部特征（时域 + 频域）
    PowerFreqFeatures extractFeatures(const std::vector<double>& signal, double samplingFreqHz);

    // v22 决策树分类
    ClassifyResult classifyFault(const PowerFreqFeatures& features);

    // ===== 评分函数 =====
    double calculateDisturbanceScore(const PowerFreqFeatures& f);
    double calculateTripScore(const PowerFreqFeatures& f);
    double calculatePowerOffSagScore(const PowerFreqFeatures& f);
    double calculateClosingSuccessScore(const PowerFreqFeatures& f);
    double calculateClosingFailureScore(const PowerFreqFeatures& f);
    double calculateSurgeScore(const PowerFreqFeatures& f);
    double calculatePowerOnScore(const PowerFreqFeatures& f);

    // ===== 信号处理辅助函数 =====

    // 计算RMS值（有效值）
    double calculateRMS(const std::vector<double>& samples);

    // 寻找周期起始过零点
    int findCycleStart(const std::vector<double>& signal, double samplingFreqHz, double targetFreq = 50.0);

    // 计算周期对齐的RMS序列（含中值滤波平滑）
    std::vector<double> computeRmsAligned(const std::vector<double>& signal, double samplingFreqHz,
                                           double targetFreq = 50.0, bool smoothRms = true);

    // 科学分段算法
    struct SegmentResult {
        int num_segments;
        std::vector<double> segment_rms_values;
        std::vector<int> segment_indices;
    };
    SegmentResult detectSegments(const std::vector<double>& rmsValues,
                                  double changeThreshold = 0.3, int minSegmentLen = -1);

    // 中值滤波
    std::vector<double> medianFilter(const std::vector<double>& data, int windowSize);

    // 计算中值
    static double computeMedian(std::vector<double> values);

    // 将原始数据转换为实际电流值
    std::vector<double> convertToActualValues(const PowerFreqCurrentData& data);

    // 获取事件类型的字符串描述
    static const char* getEventTypeName(TripEventType type);

private:
    // 提取时域特征
    PowerFreqFeatures extractTimeDomainFeatures(const std::vector<double>& signal, double samplingFreqHz);

    // 提取频域特征
    void extractFrequencyFeatures(const std::vector<double>& signal, double samplingFreqHz,
                                   PowerFreqFeatures& features);
};

#endif // POWER_FREQ_ANALYZER_H
