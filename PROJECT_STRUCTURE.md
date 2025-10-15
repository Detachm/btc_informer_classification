# 🏗️ Bitcoin Informer Classification Project Structure

## 📁 项目目录结构

```
bitcoin_informer_classification/
├── 📁 layers/                          # Transformer层实现
│   ├── __init__.py
│   ├── AutoCorrelation.py             # 自相关机制
│   ├── Autoformer_EncDec.py           # Autoformer编码解码器
│   ├── Conv_Blocks.py                 # 卷积块
│   ├── Crossformer_EncDec.py          # Crossformer编码解码器
│   ├── DWT_Decomposition.py           # 离散小波变换
│   ├── Embed.py                       # 嵌入层
│   ├── ETSformer_EncDec.py           # ETSformer编码解码器
│   ├── FourierCorrelation.py          # 傅里叶相关
│   ├── MultiWaveletCorrelation.py     # 多小波相关
│   ├── Pyraformer_EncDec.py          # Pyraformer编码解码器
│   ├── SelfAttention_Family.py       # 自注意力机制族
│   ├── StandardNorm.py                # 标准化
│   └── Transformer_EncDec.py          # Transformer编码解码器
│
├── 📁 checkpoints/                    # 模型检查点
│   ├── bitcoin_mega_classification/   # 原始模型
│   │   ├── mega_best_model.pth
│   │   └── mega_experiment_config.json
│   └── student_distillation/          # 学生模型
│       ├── student_best_model.pth
│       └── student_training_report.json
│
├── 📁 dataset/                        # 数据集
│   └── bitcoin/                       # 比特币数据
│
# archived_files 目录已清理，不再包含冗余文件
│
├── 📄 核心模型文件
├── InformerClassification.py          # 主要模型类
├── knowledge_distillation_training.py # 知识蒸馏训练
├── bitcoin_optimized_features.py     # 特征工程
│
├── 📄 训练和推理脚本
├── train_mega_bitcoin_classification.py # 原始模型训练
├── mega_realtime_prediction.py       # 实时预测
├── export_to_onnx.py                 # ONNX导出
├── rebuild_tensorrt.py               # TensorRT重建
│
├── 📄 测试和对比脚本
├── test_pytorch_models.py            # PyTorch模型测试
├── enhanced_model_comparison.py      # 增强对比测试
├── four_model_summary.py             # 性能汇总报告
├── run_onnx.py                       # ONNX推理
│
├── 📄 模型文件
├── informer_cls.fp32.onnx            # 原始模型ONNX
├── student_model.fp32.onnx           # 学生模型ONNX
├── *.trt.*.engine                   # TensorRT引擎文件
│
├── 📄 文档
├── README.md                         # 项目说明
├── 使用说明文档.md                    # 详细使用指南
├── 完整技术报告.md                    # 技术报告
├── 混合精度实际效果分析报告.md         # 混合精度分析
├── 量化敏感度评估方法.md              # 量化评估
├── PROJECT_STRUCTURE.md              # 项目结构说明
│
└── 📄 配置文件
    ├── requirements.txt              # Python依赖
    └── *.json                       # 配置文件
```

## 🔧 核心组件说明

### 1. 模型架构 (`InformerClassification.py`)
- **InformerClassification**: 主要分类模型类
- **InformerConfig**: 模型配置类
- **create_informer_classification_model**: 模型创建函数

### 2. 知识蒸馏 (`knowledge_distillation_training.py`)
- **StudentInformerClassification**: 学生模型类
- **StudentInformerConfig**: 学生模型配置
- **distillation_training**: 蒸馏训练函数

### 3. 特征工程 (`bitcoin_optimized_features.py`)
- **BitcoinOptimizedFeatureEngineer**: 特征工程类
- 包含技术指标计算、数据预处理等功能

### 4. 推理引擎
- **ONNX推理**: `run_onnx.py`
- **TensorRT推理**: `rebuild_tensorrt.py`
- **实时预测**: `mega_realtime_prediction.py`

## 🚀 快速开始

### 1. 环境安装
```bash
pip install -r requirements.txt
```

### 2. 模型测试
```bash
# 测试PyTorch模型
python test_pytorch_models.py

# 测试ONNX模型
python run_onnx.py

# 测试TensorRT引擎
python rebuild_tensorrt.py --onnx_path student_model.fp32.onnx --engine_path student_model.trt.fp32.engine --precision fp32
```

### 3. 性能对比
```bash
# 生成完整性能报告
python four_model_summary.py
```

## 📊 模型性能

| 模型类型 | 推理延迟 | 文件大小 | 加速比 | 适用场景 |
|---------|---------|---------|--------|----------|
| 学生模型_TensorRT_FP32 | 0.280ms | 2.90MB | 17.53x | 高频交易 |
| 原始模型_TensorRT_FP32 | 0.530ms | 18.62MB | 9.44x | 高性能推理 |
| 学生模型_PyTorch | 2.288ms | - | 2.21x | 开发调试 |
| 原始模型_PyTorch | 5.052ms | - | 1.00x | 基准测试 |

## 🔧 依赖关系

### 核心依赖
- **PyTorch**: 深度学习框架
- **NumPy**: 数值计算
- **Pandas**: 数据处理
- **ONNX**: 模型格式转换
- **TensorRT**: GPU推理优化

### 可选依赖
- **TensorRT**: NVIDIA GPU优化（需要CUDA）
- **Jupyter**: 交互式开发
- **Matplotlib**: 可视化

## 📝 注意事项

1. **独立性**: 项目已完全独立，不依赖外部Time-Series-Library
2. **GPU支持**: TensorRT需要NVIDIA GPU和CUDA
3. **Python版本**: 建议Python 3.8+
4. **内存要求**: 建议8GB+ RAM，4GB+ GPU内存

## 🎯 项目特点

- ✅ **完全独立**: 无外部依赖
- ✅ **多格式支持**: PyTorch/ONNX/TensorRT
- ✅ **高性能**: 17.53x加速比
- ✅ **小体积**: 90%模型压缩
- ✅ **生产就绪**: 完整的测试和文档
