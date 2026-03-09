#ifndef WAVEFORM_DATA_H
#define WAVEFORM_DATA_H

#include <vector>
#include <cstdint>
#include <string>

// 工频电流数据结构
struct PowerFreqCurrentData {
    std::vector<short> samples;       // 采样点数据（原始ADC值）
    double samplingFreqHz;            // 采样频率（Hz），如10kHz
    double ratioCoefficient;          // 变比系数（实际值 = 点位值 × ratioCoefficient ÷ 1e8）
    int64_t firstSampleTimestampNs;   // 第一个采样点的时间戳（纳秒）
    
    // 辅助函数：将原始采样值转换为实际电流值
    double convertToActualCurrent(short rawValue) const {
        return static_cast<double>(rawValue) * ratioCoefficient / 1e8;
    }
};

// 行波电流数据结构
struct TravelingWaveData {
    std::vector<short> samples;       // 采样点数据（原始ADC值）
    double samplingFreqHz;            // 采样频率（Hz），通常为200kHz~1MHz
    double ratioCoefficient;          // 变比系数（实际值 = 点位值 × ratioCoefficient ÷ 1e8）
    int64_t firstSampleTimestampNs;   // 第一个采样点的时间戳（纳秒）
    int channelId;                    // 通道标识（如A相、B相、C相）
    
    // 辅助函数：将原始采样值转换为实际电流值
    double convertToActualCurrent(short rawValue) const {
        return static_cast<double>(rawValue) * ratioCoefficient / 1e8;
    }
};

// 线路段信息（用于双端定位）
enum class LineSegmentType {
    OVERHEAD,  // 架空线
    CABLE      // 电缆
};

struct LineSegment {
    LineSegmentType type;             // 线路类型
    double lengthM;                   // 段长度（米）
    double waveSpeedMPerUs;           // 波速（m/μs），架空线约300，电缆约150-170
};

// 双端线路配置（两个采集终端之间的完整线路）
struct DualEndLineConfig {
    std::vector<LineSegment> segments;  // 从A端到B端的线路段序列
    double totalLengthM;                 // 总长度（米）
    
    // 计算总长度
    void calculateTotalLength() {
        totalLengthM = 0.0;
        for (const auto& seg : segments) {
            totalLengthM += seg.lengthM;
        }
    }
};

// 定位结果
struct LocationResult {
    double faultDistanceFromEndA_M;   // 故障点距离A端的距离（米）
    int faultSegmentIndex;             // 故障所在段索引（-1表示未确定）
    double confidenceScore;            // 置信度（0-1）
    bool isValid;                      // 结果是否有效
};

// ========== 多装置定位扩展 ==========

// 采集装置信息
struct DeviceInfo {
    std::string deviceId;              // 装置唯一标识
    double positionFromLineStart_M;    // 装置在线路上的位置（距线路起点的距离，米）
    std::string deviceName;            // 装置名称（可选）
};

// 带装置信息的行波数据
struct DeviceTravelingWaveData {
    DeviceInfo device;                 // 装置信息
    TravelingWaveData waveData;        // 行波数据
};

// 完整线路配置（包含装置位置）
struct FullLineConfig {
    std::vector<LineSegment> segments;     // 线路段序列
    std::vector<DeviceInfo> devices;       // 安装的装置列表
    double totalLengthM;                    // 总长度（米）
    
    // 计算总长度
    void calculateTotalLength() {
        totalLengthM = 0.0;
        for (const auto& seg : segments) {
            totalLengthM += seg.lengthM;
        }
    }
    
    // 验证装置位置是否在线路范围内
    bool validateDevicePositions() const {
        for (const auto& dev : devices) {
            if (dev.positionFromLineStart_M < 0 || 
                dev.positionFromLineStart_M > totalLengthM) {
                return false;
            }
        }
        return true;
    }
};

// 单次定位结果（两个装置间）
struct PairwiseLocationResult {
    std::string deviceA_Id;                // 装置A的ID
    std::string deviceB_Id;                // 装置B的ID
    double faultDistanceFromLineStart_M;   // 故障点距线路起点的距离（米）
    int faultSegmentIndex;                 // 故障所在段索引
    double confidenceScore;                // 置信度
    bool isValid;                          // 结果是否有效
};

// 多装置联合定位结果
struct MultiDeviceLocationResult {
    std::vector<PairwiseLocationResult> pairwiseResults;  // 所有配对的定位结果
    double faultDistanceFromLineStart_M;   // 综合定位结果（距线路起点，米）
    int faultSegmentIndex;                 // 故障所在段索引
    double confidenceScore;                // 综合置信度
    bool isValid;                          // 结果是否有效
    int numValidPairs;                     // 有效配对数量
};

#endif // WAVEFORM_DATA_H
