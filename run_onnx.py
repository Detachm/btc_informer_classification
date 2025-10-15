#!/usr/bin/env python3
"""
运行ONNX优化过的模型 - 简单可靠
"""

import os
import time
import numpy as np
import onnxruntime as ort

class ONNXRunner:
    def __init__(self, model_path):
        """初始化ONNX推理器"""
        self.model_path = model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到ONNX模型文件: {model_path}")
        
        self._load_model()
    
    def _load_model(self):
        """加载ONNX模型"""
        print(f"正在加载ONNX模型: {self.model_path}")
        
        # 创建推理会话，优先使用CUDA
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]
        
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            print(f"✅ ONNX模型加载成功!")
            print(f"   文件大小: {os.path.getsize(self.model_path) / 1024 / 1024:.2f} MB")
            print(f"   提供程序: {self.session.get_providers()}")
        except Exception as e:
            print(f"⚠️  CUDA加载失败，使用CPU: {e}")
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            print(f"✅ ONNX模型加载成功 (CPU)!")
    
    def predict(self, input_data):
        """执行推理预测"""
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)
        
        # 确保数据类型和形状正确
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        expected_shape = (1, 30, 33)  # batch_size, seq_len, features
        if input_data.shape != expected_shape:
            if input_data.ndim == 2:  # 如果是 (30, 33)，添加batch维度
                input_data = input_data.reshape(1, 30, 33)
            else:
                raise ValueError(f"输入形状错误，期望 {expected_shape}，得到 {input_data.shape}")
        
        # 执行推理
        start_time = time.time()
        result = self.session.run(['up_probability'], {'input': input_data})
        end_time = time.time()
        
        inference_time = end_time - start_time
        prediction = result[0][0]  # 获取预测概率
        
        return prediction, inference_time
    
    def predict_batch(self, input_data_list):
        """批量预测"""
        results = []
        total_time = 0
        
        for i, data in enumerate(input_data_list):
            prediction, inference_time = self.predict(data)
            results.append(prediction)
            total_time += inference_time
            
            if i == 0:  # 第一次预测通常较慢（包含初始化）
                print(f"第一次推理时间: {inference_time*1000:.2f} ms")
        
        avg_time = total_time / len(input_data_list)
        return results, avg_time

def main():
    """主函数 - 演示如何使用ONNX模型"""
    print("=" * 60)
    print("ONNX模型推理演示")
    print("=" * 60)
    
    # 检查可用的模型文件
    optimized_model = "informer_cls.optimized.onnx"
    original_model = "informer_cls.fp32.onnx"
    
    model_path = None
    if os.path.exists(optimized_model):
        model_path = optimized_model
        print(f"使用优化模型: {optimized_model}")
    elif os.path.exists(original_model):
        model_path = original_model
        print(f"使用原始模型: {original_model}")
    else:
        print("❌ 找不到ONNX模型文件!")
        print("请先运行: python export_to_onnx.py")
        return
    
    try:
        # 初始化推理器
        runner = ONNXRunner(model_path)
        
        # 1. 单次预测
        print(f"\n📊 单次预测测试:")
        input_data = np.random.randn(1, 30, 33).astype(np.float32)
        prediction, inference_time = runner.predict(input_data)
        
        print(f"   输入形状: {input_data.shape}")
        print(f"   推理时间: {inference_time*1000:.2f} ms")
        print(f"   预测结果: {prediction:.6f}")
        print(f"   预测类别: {'上涨' if prediction > 0.5 else '下跌'}")
        
        # 2. 批量预测
        print(f"\n📈 批量预测测试:")
        batch_size = 5
        batch_data = [np.random.randn(1, 30, 33).astype(np.float32) for _ in range(batch_size)]
        
        predictions, avg_time = runner.predict_batch(batch_data)
        
        print(f"   批量大小: {batch_size}")
        print(f"   平均推理时间: {avg_time*1000:.2f} ms")
        print(f"   总吞吐量: {batch_size/avg_time:.1f} samples/s")
        print(f"   预测结果: {[f'{p:.4f}' for p in predictions]}")
        
        # 3. 性能对比
        print(f"\n⚡ 性能总结:")
        print(f"   单次推理: {inference_time*1000:.2f} ms")
        print(f"   平均推理: {avg_time*1000:.2f} ms")
        print(f"   吞吐量: {1/avg_time:.1f} samples/s")
        
        # 4. 不同batch size测试
        print(f"\n🔢 不同batch size性能测试:")
        for bs in [1, 2, 4, 8, 16]:
            try:
                batch_data = [np.random.randn(1, 30, 33).astype(np.float32) for _ in range(bs)]
                predictions, avg_time = runner.predict_batch(batch_data)
                throughput = bs / avg_time
                print(f"   Batch {bs:2d}: {avg_time*1000:6.2f} ms, {throughput:6.1f} samples/s")
            except Exception as e:
                print(f"   Batch {bs:2d}: 失败 - {e}")
        
        print(f"\n🎉 ONNX推理演示完成!")
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()

def predict_single(input_data):
    """便捷函数：单次预测"""
    model_path = "informer_cls.optimized.onnx"
    if not os.path.exists(model_path):
        model_path = "informer_cls.fp32.onnx"
    
    runner = ONNXRunner(model_path)
    prediction, inference_time = runner.predict(input_data)
    return prediction, inference_time

def predict_from_array(data_array):
    """从numpy数组预测"""
    if data_array.shape == (30, 33):
        # 添加batch维度
        input_data = data_array.reshape(1, 30, 33)
    elif data_array.shape == (1, 30, 33):
        input_data = data_array
    else:
        raise ValueError(f"输入形状错误，期望 (30, 33) 或 (1, 30, 33)，得到 {data_array.shape}")
    
    prediction, inference_time = predict_single(input_data)
    return prediction, inference_time

if __name__ == '__main__':
    main()
