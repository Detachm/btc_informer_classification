#!/usr/bin/env python3
"""
测试原始模型和学生模型的PyTorch版本
获取基准性能数据
"""

import torch
import numpy as np
import time
import os
import warnings
import gc
warnings.filterwarnings('ignore')

# 导入项目模块
from InformerClassification import InformerClassification, InformerConfig, create_informer_classification_model
from knowledge_distillation_training import StudentInformerClassification, StudentInformerConfig

def clear_cuda_memory():
    """安全清理CUDA内存"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        print(f"CUDA内存清理警告: {e}")
    gc.collect()

def load_teacher_model(device='cuda'):
    """加载原始教师模型"""
    print("📚 加载原始教师模型...")
    teacher_model_path = 'checkpoints/bitcoin_mega_classification/mega_best_model.pth'
    
    if not os.path.exists(teacher_model_path):
        print(f"❌ 教师模型文件不存在: {teacher_model_path}")
        return None
    
    try:
        # 清理内存
        clear_cuda_memory()
        
        # 加载检查点
        checkpoint = torch.load(teacher_model_path, map_location=device)
        config_dict = checkpoint['config']
        
        # 创建配置
        teacher_config = InformerConfig()
        for key, value in config_dict.items():
            if hasattr(teacher_config, key):
                setattr(teacher_config, key, value)
        
        # 创建模型
        teacher_model, _ = create_informer_classification_model(teacher_config)
        teacher_model = teacher_model.to(device)
        
        # 处理动态投影层
        model_state_dict = checkpoint['model_state_dict']
        if 'projection.weight' in model_state_dict:
            dummy_input = torch.randn(1, teacher_config.seq_len, teacher_config.enc_in).to(device)
            with torch.no_grad():
                _ = teacher_model(dummy_input)
        
        teacher_model.load_state_dict(model_state_dict, strict=False)
        teacher_model.eval()
        
        # 计算参数量
        total_params = sum(p.numel() for p in teacher_model.parameters())
        print(f"   ✅ 教师模型加载成功")
        print(f"   参数量: {total_params:,} ({total_params/1_000_000:.1f}M)")
        
        return teacher_model
        
    except Exception as e:
        print(f"❌ 教师模型加载失败: {e}")
        clear_cuda_memory()
        return None

def load_student_model(device='cuda'):
    """加载学生模型"""
    print("🎓 加载学生模型...")
    student_model_path = 'checkpoints/student_distillation/student_best_model.pth'
    
    if not os.path.exists(student_model_path):
        print(f"❌ 学生模型文件不存在: {student_model_path}")
        return None
    
    try:
        # 清理内存
        clear_cuda_memory()
        
        # 加载检查点 - 使用CPU加载避免CUDA错误
        checkpoint = torch.load(student_model_path, map_location='cpu')
        config_dict = checkpoint['config']
        
        # 创建学生配置
        student_config = StudentInformerConfig()
        for key, value in config_dict.items():
            if hasattr(student_config, key):
                setattr(student_config, key, value)
        
        # 创建学生模型
        student_model = StudentInformerClassification(student_config)
        
        # 处理动态投影层（在CPU上）
        model_state_dict = checkpoint['model_state_dict']
        if 'projection.weight' in model_state_dict:
            dummy_input = torch.randn(1, student_config.seq_len, student_config.enc_in)
            with torch.no_grad():
                _ = student_model(dummy_input)
        
        # 加载权重
        student_model.load_state_dict(model_state_dict, strict=False)
        student_model.eval()
        
        # 移动到GPU（如果需要）
        if device == 'cuda':
            student_model = student_model.to(device)
            clear_cuda_memory()
    
        # 计算参数量
        total_params = sum(p.numel() for p in student_model.parameters())
        print(f"   ✅ 学生模型加载成功")
        print(f"   参数量: {total_params:,} ({total_params/1_000_000:.1f}M)")
        
        return student_model
        
    except Exception as e:
        print(f"❌ 学生模型加载失败: {e}")
        clear_cuda_memory()
        return None

def test_pytorch_model(model, test_input, model_name, num_runs=200):
    """测试PyTorch模型推理延迟"""
    print(f"\n🧪 测试 {model_name} (PyTorch)...")
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_input)
    
    # 精确测量
    times = []
    with torch.no_grad():
        for i in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.time()
            output = model(test_input)
            up_prob = output['up_probability']
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.time()
            times.append((end - start) * 1000)  # 转换为毫秒
    
    times_array = np.array(times)
    avg_time = np.mean(times_array)
    std_time = np.std(times_array)
    
    print(f"   ✅ {model_name} 测试完成")
    print(f"   平均推理时间: {avg_time:.3f} ± {std_time:.3f} ms")
    print(f"   输出值: {up_prob[0].item():.6f}")
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'output': up_prob[0].item(),
        'times': times_array
    }

def main():
    """主函数 - 测试PyTorch模型性能"""
    print("=" * 80)
    print("🚀 PyTorch模型性能测试 - 获取基准性能数据")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 清理GPU内存
    clear_cuda_memory()
    
    # 准备固定输入数据
    print("\n📊 准备固定输入数据...")
    test_input = torch.randn(1, 30, 33).to(device)
    print(f"输入形状: {test_input.shape}")
    
    results = {}
    
    # 1. 测试原始教师模型 (PyTorch)
    print("\n" + "="*60)
    print("1️⃣ 原始教师模型 (PyTorch)")
    print("="*60)
    teacher_model = load_teacher_model(device)
    if teacher_model is not None:
        results['原始模型_PyTorch'] = test_pytorch_model(teacher_model, test_input, "原始教师模型")
        del teacher_model
        clear_cuda_memory()
    else:
        print("⚠️ 跳过原始教师模型测试")
    
    # 2. 测试学生模型 (PyTorch)
    print("\n" + "="*60)
    print("2️⃣ 学生模型 (PyTorch)")
    print("="*60)
    student_model = load_student_model(device)
    if student_model is not None:
        results['学生模型_PyTorch'] = test_pytorch_model(student_model, test_input, "学生模型")
        del student_model
        clear_cuda_memory()
    else:
        print("⚠️ 跳过学生模型测试")
    
    # 生成对比报告
    print("\n" + "="*80)
    print("📊 PyTorch模型性能对比结果")
    print("="*80)
    
    if results:
        # 过滤掉None结果并按推理时间排序
        valid_results = {k: v for k, v in results.items() if v is not None}
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['avg_time'])
        
        print(f"{'排名':<4} {'模型类型':<25} {'平均延迟(ms)':<15} {'标准差(ms)':<12} {'输出值':<12}")
        print("-" * 80)
        
        for i, (model_name, result) in enumerate(sorted_results, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else f"{i}."
            print(f"{medal:<4} {model_name:<25} {result['avg_time']:<15.3f} {result['std_time']:<12.3f} {result['output']:<12.6f}")
        
        # 计算加速比
        if '原始模型_PyTorch' in results and results['原始模型_PyTorch'] is not None:
            baseline_time = results['原始模型_PyTorch']['avg_time']
            print(f"\n🚀 加速比分析 (以原始模型PyTorch为基准):")
            print("-" * 60)
            
            for model_name, result in results.items():
                if model_name != '原始模型_PyTorch' and result is not None:
                    speedup = baseline_time / result['avg_time']
                    print(f"{model_name:<30}: {speedup:.2f}x")
        
        # 推荐方案
        print(f"\n🎯 PyTorch模型推荐:")
        fastest_model = sorted_results[0]
        print(f"   🏆 最快推理: {fastest_model[0]} ({fastest_model[1]['avg_time']:.3f} ms)")
        
        # 学生模型信息
        if '学生模型_PyTorch' in results and results['学生模型_PyTorch'] is not None:
            student_result = results['学生模型_PyTorch']
            print(f"   🎓 学生模型: {student_result['avg_time']:.3f} ms")
    
    print(f"\n✅ PyTorch模型性能测试完成!")
    print(f"   测试次数: 200次取平均")
    print(f"   输入数据: 固定随机输入 (1, 30, 33)")
    print(f"   测试设备: {device}")
    
    # 下一步指导
    print(f"\n📋 下一步操作:")
    print(f"   1. 测试TensorRT引擎: python rebuild_tensorrt.py --onnx_path informer_cls.fp32.onnx --engine_path informer_cls.trt.fp32.engine --precision fp32")
    print(f"   2. 测试学生模型TensorRT: python rebuild_tensorrt.py --onnx_path student_model.fp32.onnx --engine_path student_model.trt.fp32.engine --precision fp32")

if __name__ == "__main__":
    main()
