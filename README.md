# Bitcoin Mega Informer 分类预测系统

基于超大型Informer模型的比特币价格方向二分类预测系统，使用4.0M参数的大模型进行实时预测。

## 🚀 系统特点

- **超大型模型**: 4.0M参数 (百万级参数量)
- **丰富特征**: 33个技术指标特征
- **实时预测**: 平均延迟仅11.86ms
- **增强架构**: 4层编码器，256维模型
- **完整数据**: 使用460万条比特币历史数据训练

## 📁 文件结构

```
bitcoin_informer_classification/
├── README.md                                    # 本文件
├── InformerClassification.py                   # 核心模型实现
├── bitcoin_optimized_features.py               # 增强特征工程
├── train_mega_bitcoin_classification.py        # 超大型模型训练
├── mega_realtime_prediction.py                 # 实时预测系统
├── checkpoints/                                # 模型检查点
│   └── bitcoin_mega_classification/            # 超大型模型
│       ├── mega_best_model.pth                # 最佳模型权重
│       └── mega_experiment_config.json        # 实验配置
├── mega_prediction_cache.pkl                  # 预测数据缓存
└── dataset/                                    # 数据目录（软链接）
    └── bitcoin/
        └── Bitcoin_BTCUSDT.csv                 # 原始比特币数据
```

## 🏗️ 模型架构

### 超大型配置
- **模型维度**: 256 (比原始模型增加4倍)
- **编码器层数**: 4层 (比原始模型增加2倍)
- **注意力头数**: 16个 (比原始模型增加2倍)
- **前馈网络**: 1024维 (比原始模型增加4倍)
- **参数量**: 3,978,243 (3.98M)

### 增强特征 (33个)
- **核心特征**: 8个 (价格比率、位置特征)
- **技术指标**: 12个 (MACD、RSI、布林带等)
- **波动率特征**: 3个 (历史波动率、ATR)
- **动量特征**: 3个 (价格动量、ROC)
- **成交量特征**: 3个 (成交量比率、OBV)
- **形态特征**: 4个 (支撑阻力位、突破信号)

## 🚀 快速开始

### 1. 训练超大型模型
```bash
python train_mega_bitcoin_classification.py
```

### 2. 实时预测
```bash
python mega_realtime_prediction.py
```

## 📊 性能指标

### 模型性能
- **参数量**: 3.98M (百万级)
- **训练数据**: 460万条记录
- **特征维度**: 33个技术指标
- **序列长度**: 30分钟历史数据

### 预测性能
- **平均延迟**: 11.86ms
- **中位数延迟**: 11.65ms
- **95th percentile**: 17.20ms
- **99th percentile**: 19.11ms

### 准确率
- **预测准确率**: ~50% (接近随机水平，符合金融预测特点)
- **预测倾向**: 略微偏向BUY信号
- **概率范围**: 0.54-0.55 (较为保守)

## 🔧 依赖要求

```bash
pip install -r requirements.txt
```

## 💡 使用说明

### ✅ 项目独立性
本项目已完全独立，不依赖外部Time-Series-Library。所有必要的模块（layers、utils）已包含在项目中。

### ⚠️ 重要说明
由于TensorRT引擎存在内存访问问题，无法在一个脚本中同时测试所有四种模型。建议使用分步测试方法。

### 快速开始
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 测试PyTorch模型
python test_pytorch_models.py

# 3. 测试ONNX模型
python run_onnx.py

# 4. 手动测试TensorRT引擎（每次只测试一个）
python rebuild_tensorrt.py --onnx_path informer_cls.fp32.onnx --engine_path informer_cls.trt.fp32.engine --precision fp32
python rebuild_tensorrt.py --onnx_path student_model.fp32.onnx --engine_path student_model.trt.fp32.engine --precision fp32

# 5. 生成性能汇总报告
python four_model_summary.py
```

### 训练模型
1. 确保数据文件路径正确
2. 运行训练脚本
3. 模型将自动保存到 `checkpoints/bitcoin_mega_classification/`

### 实时预测
1. 确保已训练好模型
2. 运行预测脚本
3. 系统会自动进行200次批量预测演示

### 缓存机制
- 首次运行会创建特征缓存
- 后续运行会从缓存快速加载
- 缓存文件: `mega_prediction_cache.pkl`

## 📈 技术优势

1. **大规模模型**: 4M参数的学习能力
2. **丰富特征**: 33个技术指标覆盖全面
3. **低延迟**: 毫秒级预测响应
4. **完整数据**: 使用完整历史数据训练
5. **自动优化**: 动态维度计算和缓存机制

## 🎯 应用场景

- **高频交易**: 毫秒级延迟适合实时交易
- **量化分析**: 丰富的技术指标特征
- **风险管理**: 保守的预测概率
- **策略回测**: 完整的历史数据支持

## 🔄 版本历史

- **v1.0**: 基础Informer模型 (249K参数)
- **v2.0**: 增强特征模型 (857K参数)
- **v3.0**: 超大型模型 (3.98M参数) ⭐

---

**注意**: 本系统仅用于研究和学习目的，不构成投资建议。金融预测具有高风险，请谨慎使用。# btc_informer_classification
