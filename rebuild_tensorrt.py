#!/usr/bin/env python3
"""
重新构建TensorRT引擎 - 使用更保守的设置
"""

import os
import tensorrt as trt
import numpy as np

def rebuild_tensorrt_engine(onnx_path="informer_cls.optimized.onnx", engine_path="informer_cls.trt.fp32.engine", precision="fp32"):
    """重新构建TensorRT引擎，使用更保守的设置"""
    print("=" * 60)
    print("重新构建TensorRT引擎")
    print("=" * 60)
    
    print(f"输入ONNX: {onnx_path}")
    print(f"输出引擎: {engine_path}")
    print(f"精度模式: {precision}")
    
    if not os.path.exists(onnx_path):
        print(f"❌ 找不到ONNX模型: {onnx_path}")
        return False
    
    try:
        # 1. 创建TensorRT日志记录器
        print("1. 创建TensorRT日志记录器...")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # 2. 创建构建器和网络
        print("2. 创建构建器和网络...")
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        # 3. 创建ONNX解析器
        print("3. 创建ONNX解析器...")
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # 4. 解析ONNX模型
        print("4. 解析ONNX模型...")
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("❌ ONNX解析失败:")
                for error in range(parser.num_errors):
                    print(f"   {parser.get_error(error)}")
                return False
        
        print("✅ ONNX解析成功!")
        
        # 5. 创建构建配置
        print("5. 创建构建配置...")
        config = builder.create_builder_config()
        
        # 设置内存池限制（TensorRT 10.x）
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        # 设置精度模式
        if precision == "fp16":
            print("   使用FP16精度...")
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            print("   使用INT8精度...")
            config.set_flag(trt.BuilderFlag.INT8)
            # 这里可以添加校准数据集
        else:
            print("   使用FP32精度...")
        
        # 6. 设置优化配置文件
        print("6. 设置优化配置文件...")
        profile = builder.create_optimization_profile()
        
        # 设置输入形状范围
        input_name = 'input'
        profile.set_shape(input_name, (1, 30, 33), (1, 30, 33), (1, 30, 33))  # 固定batch=1
        config.add_optimization_profile(profile)
        
        print(f"   输入形状: (1, 30, 33)")
        
        # 7. 构建引擎
        print("7. 构建TensorRT引擎...")
        print("   这可能需要几分钟...")
        
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print("❌ 引擎构建失败")
            return False
        
        # 8. 保存引擎
        print("8. 保存引擎...")
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        print(f"✅ 引擎构建成功!")
        print(f"   文件路径: {os.path.abspath(engine_path)}")
        print(f"   文件大小: {os.path.getsize(engine_path) / 1024 / 1024:.2f} MB")
        
        # 9. 验证引擎
        print("\n9. 验证引擎...")
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        if engine is None:
            print("❌ 引擎验证失败")
            return False
        
        print(f"✅ 引擎验证成功!")
        print(f"   IO张量数量: {engine.num_io_tensors}")
        print(f"   层数: {engine.num_layers}")
        
        # 显示张量信息
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            
            print(f"   {i}: {name} - {mode} - 形状: {shape} - 类型: {dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ 构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_new_engine(engine_path="informer_cls.trt.fp32.engine"):
    """测试新构建的引擎"""
    print(f"\n🧪 测试新构建的引擎:")
    
    if not os.path.exists(engine_path):
        print(f"❌ 找不到引擎文件: {engine_path}")
        return False
    
    try:
        # 加载引擎
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            print("❌ 引擎加载失败")
            return False
        
        print("✅ 引擎加载成功!")
        
        # 创建执行上下文
        context = engine.create_execution_context()
        if context is None:
            print("❌ 无法创建执行上下文")
            return False
        
        print("✅ 执行上下文创建成功!")
        
        # 准备测试数据
        input_data = np.random.randn(1, 30, 33).astype(np.float32)
        context.set_input_shape('input', input_data.shape)
        
        # 获取输出形状
        output_shape = context.get_tensor_shape('up_probability')
        output_data = np.empty(output_shape, dtype=np.float32)
        
        print(f"   输入形状: {input_data.shape}")
        print(f"   输出形状: {output_shape}")
        
        # 设置张量地址
        context.set_tensor_address('input', input_data.ctypes.data)
        context.set_tensor_address('up_probability', output_data.ctypes.data)
        
        # 执行推理
        import time
        start_time = time.time()
        success = context.execute_async_v3(0)
        end_time = time.time()
        
        if not success:
            print("❌ 推理失败")
            return False
        
        inference_time = end_time - start_time
        print(f"✅ 推理成功!")
        print(f"   推理时间: {inference_time*1000:.2f} ms")
        print(f"   输出值: {output_data[0]:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='构建TensorRT引擎')
    parser.add_argument('--onnx_path', type=str, default='informer_cls.optimized.onnx', help='ONNX模型路径')
    parser.add_argument('--engine_path', type=str, default='informer_cls.trt.fp32.engine', help='引擎输出路径')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16', 'int8'], help='精度模式')
    args = parser.parse_args()
    
    # 重新构建引擎
    success = rebuild_tensorrt_engine(args.onnx_path, args.engine_path, args.precision)
    
    if success:
        # 测试新引擎
        test_success = test_new_engine(args.engine_path)
        
        if test_success:
            print(f"\n🎉 TensorRT引擎重建和测试成功！")
        else:
            print(f"\n⚠️  引擎构建成功但测试失败")
    else:
        print(f"\n❌ TensorRT引擎重建失败")
    
    print("=" * 60)

if __name__ == '__main__':
    main()
