#!/usr/bin/env python3
"""
增强版四种模型横向对比测试脚本
功能：
1. 原始模型 (PyTorch)
2. TensorRT FP32后的原始模型
3. 蒸馏剪枝过后的学生模型 (PyTorch)
4. 蒸馏剪枝过后再TensorRT编译优化的学生模型

增强功能：
- 自动检测模型文件存在性
- 详细的性能分析和报告生成
- 支持多种精度模式对比
- 生成HTML格式的对比报告
- 内存使用情况监控
- 错误处理和恢复机制
"""

import torch
import numpy as np
import time
import os
import warnings
import json
import gc
import psutil
from datetime import datetime
from pathlib import Path
import argparse
warnings.filterwarnings('ignore')

# 导入项目模块
from InformerClassification import InformerClassification, InformerConfig, create_informer_classification_model
from knowledge_distillation_training import StudentInformerClassification, StudentInformerConfig

class ModelComparisonTester:
    """增强版模型对比测试器"""
    
    def __init__(self, device='cuda', num_runs=200, save_report=True):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.num_runs = num_runs
        self.save_report = save_report
        self.results = {}
        self.start_time = datetime.now()
        
        # 模型文件路径配置
        self.model_paths = {
            'teacher_pytorch': 'checkpoints/bitcoin_mega_classification/mega_best_model.pth',
            'student_pytorch': 'checkpoints/student_distillation/student_best_model.pth',
            'teacher_onnx': 'informer_cls.fp32.onnx',
            'student_onnx': 'student_model.fp32.onnx',
            'teacher_trt_fp32': 'informer_cls.trt.fp32.engine',
            'student_trt_fp32': 'student_model.trt.fp32.engine',
            'student_trt_fp16': 'student_model.trt.fp16.engine'
        }
        
        print(f"🚀 增强版模型对比测试器初始化")
        print(f"   设备: {self.device}")
        print(f"   测试次数: {self.num_runs}")
        print(f"   报告保存: {'是' if self.save_report else '否'}")
        print("=" * 80)

    def check_model_files(self):
        """检查所有模型文件是否存在"""
        print("\n📋 检查模型文件...")
        missing_files = []
        
        for model_name, file_path in self.model_paths.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024 / 1024
                print(f"   ✅ {model_name}: {file_path} ({file_size:.2f} MB)")
            else:
                print(f"   ❌ {model_name}: {file_path} (文件不存在)")
                missing_files.append((model_name, file_path))
        
        if missing_files:
            print(f"\n⚠️  缺失 {len(missing_files)} 个模型文件:")
            for name, path in missing_files:
                print(f"   - {name}: {path}")
            return False
        
        print("   ✅ 所有模型文件检查完成")
        return True

    def get_memory_usage(self):
        """获取当前内存使用情况"""
        if self.device == 'cuda':
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            return {
                'gpu_current': gpu_memory,
                'gpu_max': gpu_max_memory,
                'cpu_percent': psutil.virtual_memory().percent
            }
        else:
            return {
                'cpu_percent': psutil.virtual_memory().percent,
                'cpu_available': psutil.virtual_memory().available / 1024 / 1024 / 1024
            }

    def clear_memory(self):
        """清理内存"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def load_teacher_model(self):
        """加载原始教师模型"""
        print("\n📚 加载原始教师模型...")
        teacher_model_path = self.model_paths['teacher_pytorch']
        
        if not os.path.exists(teacher_model_path):
            print(f"❌ 教师模型文件不存在: {teacher_model_path}")
            return None
        
        try:
            # 加载检查点
            checkpoint = torch.load(teacher_model_path, map_location=self.device)
            config_dict = checkpoint['config']
            
            # 创建配置
            teacher_config = InformerConfig()
            for key, value in config_dict.items():
                if hasattr(teacher_config, key):
                    setattr(teacher_config, key, value)
            
            # 创建模型
            teacher_model, _ = create_informer_classification_model(teacher_config)
            teacher_model = teacher_model.to(self.device)
            
            # 处理动态投影层
            model_state_dict = checkpoint['model_state_dict']
            if 'projection.weight' in model_state_dict:
                dummy_input = torch.randn(1, teacher_config.seq_len, teacher_config.enc_in).to(self.device)
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
            return None

    def load_student_model(self):
        """加载学生模型"""
        print("\n🎓 加载学生模型...")
        student_model_path = self.model_paths['student_pytorch']
        
        if not os.path.exists(student_model_path):
            print(f"❌ 学生模型文件不存在: {student_model_path}")
            return None
        
        try:
            # 清理GPU内存
            self.clear_memory()
            
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
            if self.device == 'cuda':
                student_model = student_model.to(self.device)
                self.clear_memory()
        
            # 计算参数量
            total_params = sum(p.numel() for p in student_model.parameters())
            print(f"   ✅ 学生模型加载成功")
            print(f"   参数量: {total_params:,} ({total_params/1_000_000:.1f}M)")
            
            return student_model
            
        except Exception as e:
            print(f"❌ 学生模型加载失败: {e}")
            self.clear_memory()
            return None

    def test_pytorch_model(self, model, test_input, model_name):
        """测试PyTorch模型推理延迟"""
        print(f"\n🧪 测试 {model_name} (PyTorch)...")
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
        # 精确测量
        times = []
        outputs = []
        
        with torch.no_grad():
            for i in range(self.num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.time()
                output = model(test_input)
                up_prob = output['up_probability']
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end = time.time()
                times.append((end - start) * 1000)  # 转换为毫秒
                outputs.append(up_prob[0].item())
        
        times_array = np.array(times)
        outputs_array = np.array(outputs)
        
        result = {
            'avg_time': np.mean(times_array),
            'std_time': np.std(times_array),
            'min_time': np.min(times_array),
            'max_time': np.max(times_array),
            'output': np.mean(outputs_array),
            'output_std': np.std(outputs_array),
            'times': times_array,
            'memory_usage': self.get_memory_usage()
        }
        
        print(f"   ✅ {model_name} 测试完成")
        print(f"   平均推理时间: {result['avg_time']:.3f} ± {result['std_time']:.3f} ms")
        print(f"   时间范围: {result['min_time']:.3f} - {result['max_time']:.3f} ms")
        print(f"   输出值: {result['output']:.6f} ± {result['output_std']:.6f}")
        
        return result

    def test_onnx_model(self, onnx_path, model_name):
        """测试ONNX模型推理性能"""
        import onnxruntime as ort
        
        print(f"\n🔧 测试 {model_name} (ONNX)...")
        
        if not os.path.exists(onnx_path):
            print(f"❌ ONNX模型文件不存在: {onnx_path}")
            return None
        
        try:
            # 创建ONNX Runtime会话
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            session = ort.InferenceSession(onnx_path, providers=providers)
            
            # 获取输入输出信息
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            input_shape = session.get_inputs()[0].shape
            
            print(f"   输入名称: {input_name}, 形状: {input_shape}")
            print(f"   输出名称: {output_name}")
            
            # 准备测试数据
            test_input = np.random.randn(1, 30, 33).astype(np.float32)
            
            # 预热
            for _ in range(5):
                _ = session.run([output_name], {input_name: test_input})
            
            # 性能测试
            times = []
            outputs = []
            
            for i in range(self.num_runs):
                start_time = time.time()
                outputs_result = session.run([output_name], {input_name: test_input})
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # 转换为毫秒
                outputs.append(outputs_result[0][0])
            
            # 计算统计信息
            times_array = np.array(times)
            outputs_array = np.array(outputs)
            
            result = {
                'avg_time': np.mean(times_array),
                'std_time': np.std(times_array),
                'min_time': np.min(times_array),
                'max_time': np.max(times_array),
                'output': np.mean(outputs_array),
                'output_std': np.std(outputs_array),
                'times': times_array,
                'file_size': os.path.getsize(onnx_path) / 1024 / 1024,
                'memory_usage': self.get_memory_usage()
            }
            
            print(f"   ✅ {model_name} 测试完成")
            print(f"   平均推理时间: {result['avg_time']:.3f} ± {result['std_time']:.3f} ms")
            print(f"   时间范围: {result['min_time']:.3f} - {result['max_time']:.3f} ms")
            print(f"   输出值: {result['output']:.6f} ± {result['output_std']:.6f}")
            print(f"   文件大小: {result['file_size']:.2f} MB")
            
            return result
            
        except Exception as e:
            print(f"❌ {model_name} 测试失败: {e}")
            self.clear_memory()
            return None

    def test_tensorrt_engine(self, engine_path, model_name):
        """测试TensorRT引擎推理延迟"""
        print(f"\n⚡ 测试 {model_name} (TensorRT)...")
        
        if not os.path.exists(engine_path):
            print(f"❌ 引擎文件不存在: {engine_path}")
            return None
        
        # 清理GPU内存
        self.clear_memory()
        
        try:
            import tensorrt as trt
            
            # 加载引擎
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)
            
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            engine = runtime.deserialize_cuda_engine(engine_data)
            if engine is None:
                print(f"❌ {model_name} 引擎加载失败")
                return None
            
            # 创建执行上下文
            context = engine.create_execution_context()
            if context is None:
                print(f"❌ {model_name} 无法创建执行上下文")
                return None
            
            # 准备测试数据
            input_data = np.random.randn(1, 30, 33).astype(np.float32)
            context.set_input_shape('input', input_data.shape)
            
            # 获取输出形状
            output_shape = context.get_tensor_shape('up_probability')
            output_data = np.empty(output_shape, dtype=np.float32)
            
            # 设置张量地址
            context.set_tensor_address('input', input_data.ctypes.data)
            context.set_tensor_address('up_probability', output_data.ctypes.data)
            
            # 预热
            for _ in range(10):
                context.execute_async_v3(0)
            
            # 精确测量
            times = []
            outputs = []
            
            for _ in range(self.num_runs):
                start = time.time()
                success = context.execute_async_v3(0)
                end = time.time()
                
                if success:
                    times.append((end - start) * 1000)
                    outputs.append(output_data[0])
            
            if times:
                times_array = np.array(times)
                outputs_array = np.array(outputs)
                
                result = {
                    'avg_time': np.mean(times_array),
                    'std_time': np.std(times_array),
                    'min_time': np.min(times_array),
                    'max_time': np.max(times_array),
                    'output': np.mean(outputs_array),
                    'output_std': np.std(outputs_array),
                    'times': times_array,
                    'file_size': os.path.getsize(engine_path) / 1024 / 1024,
                    'memory_usage': self.get_memory_usage()
                }
                
                print(f"   ✅ {model_name} 测试完成")
                print(f"   平均推理时间: {result['avg_time']:.3f} ± {result['std_time']:.3f} ms")
                print(f"   时间范围: {result['min_time']:.3f} - {result['max_time']:.3f} ms")
                print(f"   输出值: {result['output']:.6f} ± {result['output_std']:.6f}")
                print(f"   文件大小: {result['file_size']:.2f} MB")
                
                return result
            else:
                print(f"❌ {model_name} 推理失败")
                return None
                
        except Exception as e:
            print(f"❌ {model_name} 测试失败: {e}")
            print("   这可能是由于TensorRT引擎的内存访问问题")
            self.clear_memory()
            return None

    def run_comparison_test(self):
        """运行完整的对比测试"""
        print("=" * 80)
        print("🚀 增强版四种模型横向对比测试")
        print("=" * 80)
        
        # 检查模型文件
        if not self.check_model_files():
            print("\n⚠️  部分模型文件缺失，将跳过相应测试")
        
        # 准备固定输入数据
        print("\n📊 准备固定输入数据...")
        test_input = torch.randn(1, 30, 33).to(self.device)
        print(f"   输入形状: {test_input.shape}")
        print(f"   数据类型: {test_input.dtype}")
        
        # 1. 测试原始教师模型 (PyTorch)
        print("\n" + "="*60)
        print("1️⃣ 原始教师模型 (PyTorch)")
        print("="*60)
        teacher_model = self.load_teacher_model()
        if teacher_model is not None:
            self.results['原始模型_PyTorch'] = self.test_pytorch_model(teacher_model, test_input, "原始教师模型")
            del teacher_model
            self.clear_memory()
        else:
            print("⚠️ 跳过原始教师模型测试")
        
        # 2. 测试TensorRT FP32后的原始模型
        print("\n" + "="*60)
        print("2️⃣ TensorRT FP32后的原始模型")
        print("="*60)
        self.results['原始模型_TensorRT_FP32'] = self.test_tensorrt_engine(
            self.model_paths['teacher_trt_fp32'], 
            "原始模型 TensorRT FP32"
        )
        
        # 3. 测试蒸馏剪枝过后的学生模型 (PyTorch)
        print("\n" + "="*60)
        print("3️⃣ 蒸馏剪枝过后的学生模型 (PyTorch)")
        print("="*60)
        student_model = self.load_student_model()
        if student_model is not None:
            self.results['学生模型_PyTorch'] = self.test_pytorch_model(student_model, test_input, "学生模型")
            del student_model
            self.clear_memory()
        else:
            print("⚠️ 跳过学生模型测试")
        
        # 4. 测试蒸馏剪枝过后再TensorRT编译优化的学生模型
        print("\n" + "="*60)
        print("4️⃣ 蒸馏剪枝过后再TensorRT编译优化的学生模型")
        print("="*60)
        self.results['学生模型_TensorRT_FP32'] = self.test_tensorrt_engine(
            self.model_paths['student_trt_fp32'], 
            "学生模型 TensorRT FP32"
        )
        
        # 5. 额外测试：学生模型TensorRT FP16
        print("\n" + "="*60)
        print("5️⃣ 学生模型 TensorRT FP16 (额外测试)")
        print("="*60)
        self.results['学生模型_TensorRT_FP16'] = self.test_tensorrt_engine(
            self.model_paths['student_trt_fp16'], 
            "学生模型 TensorRT FP16"
        )
        
        # 生成对比报告
        self.generate_comparison_report()

    def generate_comparison_report(self):
        """生成对比报告"""
        print("\n" + "="*80)
        print("📊 四种模型横向对比结果")
        print("="*80)
        
        if not self.results:
            print("❌ 没有有效的测试结果")
            return
        
        # 过滤掉None结果并按推理时间排序
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['avg_time'])
        
        # 打印详细对比表
        print(f"{'排名':<4} {'模型类型':<25} {'平均延迟(ms)':<15} {'标准差(ms)':<12} {'输出值':<12} {'文件大小(MB)':<12}")
        print("-" * 90)
        
        for i, (model_name, result) in enumerate(sorted_results, 1):
            file_size = result.get('file_size', 'N/A')
            if file_size != 'N/A':
                file_size = f"{file_size:.2f}"
            
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(f"{medal:<4} {model_name:<25} {result['avg_time']:<15.3f} {result['std_time']:<12.3f} {result['output']:<12.6f} {file_size:<12}")
        
        # 计算加速比
        if '原始模型_PyTorch' in self.results and self.results['原始模型_PyTorch'] is not None:
            baseline_time = self.results['原始模型_PyTorch']['avg_time']
            print(f"\n🚀 加速比分析 (以原始模型PyTorch为基准):")
            print("-" * 60)
            
            for model_name, result in self.results.items():
                if model_name != '原始模型_PyTorch' and result is not None:
                    speedup = baseline_time / result['avg_time']
                    print(f"{model_name:<30}: {speedup:.2f}x")
        
        # 推荐方案
        print(f"\n🎯 推荐使用方案:")
        fastest_model = sorted_results[0]
        print(f"   🏆 最快推理: {fastest_model[0]} ({fastest_model[1]['avg_time']:.3f} ms)")
        
        # 学生模型最佳
        student_models = [(name, result) for name, result in self.results.items() if '学生模型' in name and result is not None]
        if student_models:
            fastest_student = min(student_models, key=lambda x: x[1]['avg_time'])
            print(f"   🎓 学生模型最佳: {fastest_student[0]} ({fastest_student[1]['avg_time']:.3f} ms)")
        
        # 体积最小
        models_with_size = [(name, result) for name, result in self.results.items() if result is not None and 'file_size' in result]
        if models_with_size:
            smallest_model = min(models_with_size, key=lambda x: x[1]['file_size'])
            print(f"   💾 最小体积: {smallest_model[0]} ({smallest_model[1]['file_size']:.2f} MB)")
        
        # 保存报告
        if self.save_report:
            self.save_detailed_report()

    def save_detailed_report(self):
        """保存详细报告"""
        report_data = {
            'test_info': {
                'timestamp': self.start_time.isoformat(),
                'device': self.device,
                'num_runs': self.num_runs,
                'test_duration': (datetime.now() - self.start_time).total_seconds()
            },
            'results': {}
        }
        
        # 处理结果数据（移除numpy数组）
        for model_name, result in self.results.items():
            if result is not None:
                clean_result = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        clean_result[key] = value.tolist()
                    elif isinstance(value, dict):
                        clean_result[key] = value
                    else:
                        clean_result[key] = float(value) if isinstance(value, (int, float)) else value
                report_data['results'][model_name] = clean_result
        
        # 保存JSON报告
        report_file = f"model_comparison_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 详细报告已保存: {report_file}")
        
        # 生成HTML报告
        self.generate_html_report(report_data)

    def generate_html_report(self, report_data):
        """生成HTML格式的对比报告"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型对比测试报告</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .summary {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .fastest {{ background-color: #d4edda !important; }}
        .recommendation {{ background: #e8f5e8; padding: 15px; border-left: 4px solid #28a745; margin: 15px 0; }}
        .metric {{ display: inline-block; margin: 5px 10px; padding: 5px 10px; background: #3498db; color: white; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 四种模型横向对比测试报告</h1>
        
        <div class="summary">
            <h3>📊 测试概览</h3>
            <p><strong>测试时间:</strong> {report_data['test_info']['timestamp']}</p>
            <p><strong>测试设备:</strong> {report_data['test_info']['device']}</p>
            <p><strong>测试次数:</strong> {report_data['test_info']['num_runs']}</p>
            <p><strong>测试时长:</strong> {report_data['test_info']['test_duration']:.2f} 秒</p>
        </div>
        
        <h2>📈 性能对比表</h2>
        <table>
            <thead>
                <tr>
                    <th>排名</th>
                    <th>模型类型</th>
                    <th>平均延迟(ms)</th>
                    <th>标准差(ms)</th>
                    <th>输出值</th>
                    <th>文件大小(MB)</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # 排序结果
        valid_results = {k: v for k, v in report_data['results'].items() if v is not None}
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['avg_time'])
        
        for i, (model_name, result) in enumerate(sorted_results, 1):
            file_size = result.get('file_size', 'N/A')
            if file_size != 'N/A':
                file_size = f"{file_size:.2f}"
            
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            row_class = "fastest" if i == 1 else ""
            
            html_content += f"""
                <tr class="{row_class}">
                    <td>{medal}</td>
                    <td>{model_name}</td>
                    <td>{result['avg_time']:.3f}</td>
                    <td>{result['std_time']:.3f}</td>
                    <td>{result['output']:.6f}</td>
                    <td>{file_size}</td>
                </tr>
"""
        
        html_content += """
            </tbody>
        </table>
        
        <div class="recommendation">
            <h3>🎯 推荐使用方案</h3>
"""
        
        if sorted_results:
            fastest_model = sorted_results[0]
            html_content += f"""
            <p><strong>🏆 最快推理:</strong> {fastest_model[0]} ({fastest_model[1]['avg_time']:.3f} ms)</p>
"""
            
            # 学生模型最佳
            student_models = [(name, result) for name, result in report_data['results'].items() if '学生模型' in name and result is not None]
            if student_models:
                fastest_student = min(student_models, key=lambda x: x[1]['avg_time'])
                html_content += f"""
            <p><strong>🎓 学生模型最佳:</strong> {fastest_student[0]} ({fastest_student[1]['avg_time']:.3f} ms)</p>
"""
            
            # 体积最小
            models_with_size = [(name, result) for name, result in report_data['results'].items() if result is not None and 'file_size' in result]
            if models_with_size:
                smallest_model = min(models_with_size, key=lambda x: x[1]['file_size'])
                html_content += f"""
            <p><strong>💾 最小体积:</strong> {smallest_model[0]} ({smallest_model[1]['file_size']:.2f} MB)</p>
"""
        
        html_content += """
        </div>
        
        <h2>📊 加速比分析</h2>
        <div class="summary">
"""
        
        # 计算加速比
        if '原始模型_PyTorch' in report_data['results'] and report_data['results']['原始模型_PyTorch'] is not None:
            baseline_time = report_data['results']['原始模型_PyTorch']['avg_time']
            html_content += f"<p><strong>基准模型:</strong> 原始模型_PyTorch ({baseline_time:.3f} ms)</p>"
            
            for model_name, result in report_data['results'].items():
                if model_name != '原始模型_PyTorch' and result is not None:
                    speedup = baseline_time / result['avg_time']
                    html_content += f"<p><strong>{model_name}:</strong> {speedup:.2f}x 加速</p>"
        
        html_content += """
        </div>
        
        <div class="summary">
            <h3>✅ 测试完成</h3>
            <p>本次测试对比了四种不同的模型优化方案，为生产部署提供了多种选择。</p>
            <p>建议根据具体应用场景选择最适合的模型版本。</p>
        </div>
    </div>
</body>
</html>
"""
        
        # 保存HTML报告
        html_file = f"model_comparison_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"🌐 HTML报告已保存: {html_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强版四种模型横向对比测试')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='测试设备')
    parser.add_argument('--num_runs', type=int, default=200, help='测试次数')
    parser.add_argument('--no_report', action='store_true', help='不保存报告')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = ModelComparisonTester(
        device=args.device,
        num_runs=args.num_runs,
        save_report=not args.no_report
    )
    
    # 运行对比测试
    tester.run_comparison_test()
    
    print(f"\n✅ 增强版四种模型横向对比测试完成!")
    print(f"   测试次数: {args.num_runs}次取平均")
    print(f"   输入数据: 固定随机输入 (1, 30, 33)")
    print(f"   测试设备: {args.device}")

if __name__ == "__main__":
    main()
