# 🚀 Bitcoin Informer 分类模型使用说明文档

## 📋 项目概述

本项目实现了基于Informer架构的比特币价格预测分类模型，通过知识蒸馏、结构化剪枝和TensorRT优化，实现了**17.53倍的推理加速**和**90%的模型压缩**。

### 🎯 核心特性
- **超高性能**: 推理延迟低至0.280ms
- **极小体积**: 模型大小仅1.81MB
- **多精度支持**: FP32/FP16/INT8量化
- **跨平台部署**: 支持ONNX和TensorRT

## 🚀 快速开始

### 1. 测试PyTorch模型性能
```bash
python test_pytorch_models.py
```

### 2. 测试TensorRT引擎性能
```bash
# 测试原始模型TensorRT
python rebuild_tensorrt.py --onnx_path informer_cls.fp32.onnx --engine_path informer_cls.trt.fp32.engine --precision fp32

# 测试学生模型TensorRT
python rebuild_tensorrt.py --onnx_path student_model.fp32.onnx --engine_path student_model.trt.fp32.engine --precision fp32
```

### 3. 查看完整性能对比
运行上述步骤后，您将获得四种模型的完整性能对比数据。

## 🛠️ 环境要求

### 基础环境
```bash
# Python版本
Python >= 3.8

# 主要依赖
torch >= 1.12.0
tensorrt >= 10.0.0
onnx >= 1.14.0
numpy >= 1.21.0
pandas >= 1.3.0
```

### GPU环境（推荐）
```bash
# CUDA版本
CUDA >= 11.8

# GPU内存
GPU Memory >= 4GB (推荐8GB+)

# 支持的GPU
NVIDIA GPU with Compute Capability >= 6.1
```

### 安装依赖
```bash
# 创建虚拟环境
conda create -n bitcoin_informer python=3.8
conda activate bitcoin_informer

# 安装PyTorch (CUDA版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install tensorrt onnx onnxruntime-gpu pandas numpy scikit-learn matplotlib seaborn
```

## 📁 项目结构

```
bitcoin_informer_classification/
├── 📁 checkpoints/                    # 模型权重文件
│   ├── bitcoin_mega_classification/   # 教师模型
│   ├── student_distillation/          # 知识蒸馏模型
│   └── student_pruned/                # 剪枝模型
├── 📁 dataset/                        # 数据集
├── 🔧 核心脚本/
│   ├── InformerClassification.py      # 模型定义
│   ├── run_onnx.py                    # ONNX推理脚本 ⭐
│   ├── model_comparison_test.py       # 四模型延迟对比脚本 ⭐
│   ├── rebuild_tensorrt.py            # TensorRT构建脚本
│   ├── export_to_onnx.py              # ONNX导出脚本
│   ├── train_mega_bitcoin_classification.py # 模型训练脚本
│   └── mega_realtime_prediction.py    # 实时预测脚本
├── 🎯 推理引擎/
│   ├── student_model.trt.fp32.engine  # 最佳性能引擎
│   ├── student_model.trt.fp16.engine  # 最小体积引擎
│   ├── student_model.fp32.onnx        # 跨平台模型
│   ├── informer_cls.fp32.onnx         # 原始ONNX模型
│   ├── informer_cls.optimized.onnx    # 优化ONNX模型
│   └── informer_cls.trt.*.engine      # 原始TensorRT引擎
├── 📚 文档/
│   ├── 使用说明文档.md                # 本文档 ⭐
│   ├── README.md                      # 项目说明
│   ├── FINAL_GUIDE.md                 # 最终使用指南
│   ├── FOUR_MODEL_COMPARISON_REPORT.md # 四模型性能对比报告 ⭐
│   ├── 完整技术报告.md                # 技术分析报告
│   ├── 混合精度实际效果分析报告.md    # 混合精度分析
│   └── 量化敏感度评估方法.md          # 量化评估方法
└── 📁 archived_files/                 # 归档文件（已清理）
```
## 📊 模型性能对比

> 📋 详细对比数据请查看: `FOUR_MODEL_COMPARISON_REPORT.md`

| 排名 | 模型类型 | 推理时间 | 文件大小 | 加速比 | 推荐场景 |
|------|---------|---------|---------|--------|----------|
| 🥇 | **学生模型_TensorRT_FP32** | **0.280ms** | 2.81MB | **17.53x** | 高频交易、实时预测 |
| 🥈 | **学生模型_TensorRT_FP16** | 0.420ms | 1.81MB | 11.69x | 边缘设备、移动端 |
| 🥉 | 原始模型_TensorRT_FP32 | 0.520ms | 18.49MB | 9.44x | 基准对比 |
| 4 | **学生模型_ONNX** | 0.641ms | 3.33MB | 7.66x | 跨平台、云端服务 |
| 5 | 学生模型_PyTorch | 2.260ms | - | 2.17x | 开发调试 |
| 6 | 原始模型_PyTorch | 4.909ms | - | 1.00x | 基准测试 |


### 步骤1: 测试PyTorch模型

```bash
python test_pytorch_models.py
```

**说明**: 测试原始模型和学生模型的PyTorch版本，获取基准性能

**预期输出**:
```
================================================================================
🚀 PyTorch模型性能测试 - 获取基准性能数据
================================================================================
🥇 学生模型_PyTorch: 2.288 ± 0.154 ms
🥈 原始模型_PyTorch: 5.052 ± 0.406 ms
🚀 加速比: 学生模型比原始模型快 2.21x
```
    
    
### 步骤2: 重建原始模型TensorRT引擎

```bash
python rebuild_tensorrt.py --onnx_path informer_cls.fp32.onnx --engine_path informer_cls.trt.fp32.engine --precision fp32
```

**说明**: 重建原始模型的TensorRT引擎并测试性能

**预期输出**:
```
🎯 重新构建TensorRT引擎
============================================================
✅ 引擎构建成功: informer_cls.trt.fp32.engine
📊 推理测试:
   推理时间: 0.520 ms
   预测概率: 0.5000
   加速比: 9.44x
```

### 步骤3: 重建学生模型TensorRT引擎

```bash
python rebuild_tensorrt.py --onnx_path student_model.fp32.onnx --engine_path student_model.trt.fp32.engine --precision fp32
```

**说明**: 重建学生模型的TensorRT引擎并测试性能

**预期输出**:
```
🎯 重新构建TensorRT引擎
============================================================
✅ 引擎构建成功: student_model.trt.fp32.engine
📊 推理测试:
   推理时间: 0.280 ms
   预测概率: 0.5000
   加速比: 17.53x
```

### 步骤4: 可选 - 测试FP16引擎

```bash
python rebuild_tensorrt.py --onnx_path student_model.fp32.onnx --engine_path student_model.trt.fp16.engine --precision fp16
```

**说明**: 测试学生模型的FP16 TensorRT引擎，获得更小的文件体积

**预期输出**:
```
🎯 重新构建TensorRT引擎
============================================================
✅ 引擎构建成功: student_model.trt.fp16.engine
📊 推理测试:
   推理时间: 0.420 ms
   预测概率: 0.5000
   文件大小: 1.81 MB
   加速比: 11.69x
```