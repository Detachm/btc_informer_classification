#!/usr/bin/env python3
"""
超大型比特币模型实时预测器
使用训练好的3.8M参数模型进行实时预测
"""

import torch
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from InformerClassification import create_informer_classification_model, InformerConfig
from bitcoin_optimized_features import BitcoinOptimizedFeatureEngineer
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

class MegaRealtimeBitcoinPredictor:
    """超大型比特币实时预测器"""
    
    def __init__(self, model_path, scaler_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_path = model_path
        self.scaler = None
        self.model = None
        self.config = None
        self.feature_names = None
        
        # 预测历史
        self.prediction_history = deque(maxlen=100)
        self.latency_history = deque(maxlen=100)
        
        # 加载模型和数据预处理器
        self.load_model()

    # （已移除）端到端延迟测量：保留在批量预测处的简单统计，避免重复计算

    # -----------------------------
    # 算子火焰图/时间线采集（可在 TensorBoard 或 Chrome Trace 查看）
    # -----------------------------
    def profile_inference(self, data_batch, batch_size=64, steps=16, logdir='./tb_logs_realtime'):
        """采集推理算子火焰图/时间线

        生成的 trace 保存在 logdir 下，可通过以下方式查看：
          1) TensorBoard:  tensorboard --logdir ./tb_logs_realtime
          2) Chrome: chrome://tracing 打开导出的 trace.json
        """
        os.makedirs(logdir, exist_ok=True)

        # 准备一个稳定的批量输入，避免采样期间形状变化影响内核选型
        def build_fixed_inputs():
            prepared = []
            now_len = len(data_batch)
            base_end = max(self.config.seq_len, now_len)
            idxs = np.linspace(max(1, self.config.seq_len), base_end, num=batch_size, dtype=int)
            for idx in idxs:
                sl = min(idx, now_len)
                cur = data_batch[:sl]
                prepared.append(self.prepare_input(cur))
            return torch.cat(prepared, dim=0)

        inputs = build_fixed_inputs()

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        # 使用调度：等待2步（稳定）、预热2步、采集 steps 步
        sched = torch.profiler.schedule(wait=2, warmup=2, active=steps, repeat=1)

        logdir_ts = os.path.join(logdir, datetime.now().strftime('%Y%m%d_%H%M%S'))
        print(f"\n[Profiler] 日志目录: {logdir_ts}")

        with profile(
            activities=activities,
            schedule=sched,
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            on_trace_ready=tensorboard_trace_handler(logdir_ts)
        ) as prof:
            total_iters = 2 + 2 + steps
            for _ in range(total_iters):
                with torch.no_grad(), record_function("inference_step"):
                    _ = self.model(inputs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                prof.step()

        # 在控制台打印 Top 算子概览（按 CPU 自身时间排序）
        print("\n[Profiler 概览 - 前20] (按 self_cpu_time_total 排序)")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
        
    def load_model(self):
        """加载训练好的超大型模型"""
        print("正在加载超大型模型...")
        
        # 加载检查点
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 获取配置
        config_dict = checkpoint['config']
        self.config = MegaInformerConfig()
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # 获取特征名称
        self.feature_names = checkpoint['feature_names']
        print(f"特征数量: {len(self.feature_names)}")
        
        # 获取标准化器
        self.scaler = checkpoint['scaler']
        
        # 创建模型并移到设备
        self.model, _ = create_informer_classification_model(self.config)
        self.model = self.model.to(self.device)
        
        # 处理动态创建的投影层
        model_state_dict = checkpoint['model_state_dict']
        
        # 如果保存的模型包含投影层，需要先触发一次前向传播来创建投影层
        if 'projection.weight' in model_state_dict:
            # 创建一个虚拟输入来触发投影层的创建
            dummy_input = torch.randn(1, self.config.seq_len, self.config.enc_in).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        # 加载状态字典
        self.model.load_state_dict(model_state_dict, strict=False)
        self.model.eval()
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型参数量: {total_params:,} ({total_params/1_000_000:.1f}M)")
        print("✓ 超大型模型加载完成")
        
    def create_features(self, df):
        """创建增强特征"""
        engineer = BitcoinOptimizedFeatureEngineer()
        feature_names = engineer.create_optimized_features(df)
        
        # 处理缺失值
        df = df.fillna(method='bfill').fillna(method='ffill')
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # 处理异常值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in feature_names:
                upper_limit = df[col].quantile(0.999)
                lower_limit = df[col].quantile(0.001)
                df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
        
        return df[feature_names].values
    
    def prepare_input(self, data, seq_len=30):
        """准备模型输入"""
        # 标准化
        data_scaled = self.scaler.transform(data)
        
        # 创建时间序列
        if len(data_scaled) >= seq_len:
            sequence = data_scaled[-seq_len:]  # 取最后seq_len个时间点
            return torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        else:
            # 如果数据不够，用零填充
            padding = np.zeros((seq_len - len(data_scaled), data_scaled.shape[1]))
            sequence = np.vstack([padding, data_scaled])
            return torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
    
    def predict_single(self, data):
        """单次预测"""
        start_time = time.time()
        
        # 准备输入
        input_tensor = self.prepare_input(data)
        
        # 预测
        with torch.no_grad():
            output = self.model(input_tensor)
            up_prob = output['up_probability'].cpu().numpy()[0]
            prediction = 'BUY' if up_prob > 0.5 else 'SELL'
        
        # 计算延迟
        latency = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 记录历史
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'probability': up_prob,
            'latency': latency
        })
        self.latency_history.append(latency)
        
        return prediction, up_prob, latency
    
    def predict_batch(self, data_batch, num_predictions=100):
        """批量预测演示"""
        print(f"开始批量预测演示 ({num_predictions} 次)...")
        print("="*60)
        
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(num_predictions):
            # 获取当前时间点的数据
            if i + 31 < len(data_batch):  # 需要30个历史点 + 当前点
                current_data = data_batch[:i + 31]  # 前i+31个数据点
                actual_direction = '上涨' if data_batch[i + 31][3] > data_batch[i + 30][3] else '下跌'  # 比较收盘价
            else:
                current_data = data_batch[:i + 1]
                actual_direction = '未知'
            
            # 预测
            prediction, prob, latency = self.predict_single(current_data)
            
            # 计算准确率
            if actual_direction != '未知':
                is_correct = (prediction == 'BUY' and actual_direction == '上涨') or (prediction == 'SELL' and actual_direction == '下跌')
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
            
            # 显示结果
            if i % 10 == 0 or i == num_predictions - 1:
                accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
                print(f"预测 {i+1:3d}: {prediction} (p={prob:.3f}) | 实际: {actual_direction} | "
                      f"延迟: {latency:.1f}ms | 准确率: {accuracy:.1f}%")
        
        # 最终统计
        print("\n" + "="*60)
        print("批量预测结果统计")
        print("="*60)
        print(f"总预测次数: {num_predictions}")
        print(f"有效预测次数: {total_predictions}")
        print(f"预测准确率: {correct_predictions / total_predictions * 100:.2f}%" if total_predictions > 0 else "预测准确率: N/A")
        
        # 延迟统计
        if self.latency_history:
            avg_latency = np.mean(list(self.latency_history))
            median_latency = np.median(list(self.latency_history))
            p95_latency = np.percentile(list(self.latency_history), 95)
            p99_latency = np.percentile(list(self.latency_history), 99)
            
            print(f"\n延迟统计:")
            print(f"平均延迟: {avg_latency:.2f} ms")
            print(f"中位数延迟: {median_latency:.2f} ms")
            print(f"95th percentile: {p95_latency:.2f} ms")
            print(f"99th percentile: {p99_latency:.2f} ms")
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0
    
    def simulate_realtime(self, data, duration_minutes=10):
        """模拟实时预测"""
        print(f"开始实时预测模拟 ({duration_minutes} 分钟)...")
        print("="*60)
        
        start_time = time.time()
        end_time = start_time + duration_minutes * 60
        
        prediction_count = 0
        
        while time.time() < end_time:
            # 获取当前时间的数据
            current_time = datetime.now()
            current_minute = current_time.minute
            
            # 模拟每分钟预测一次
            if prediction_count == 0 or current_minute != last_minute:
                # 获取最新的数据
                latest_data = data[-31:] if len(data) >= 31 else data
                
                # 预测
                prediction, prob, latency = self.predict_single(latest_data)
                
                # 显示结果
                print(f"[{current_time.strftime('%H:%M:%S')}] {prediction} (p={prob:.3f}) | 延迟: {latency:.1f}ms")
                
                prediction_count += 1
                last_minute = current_minute
            
            time.sleep(1)  # 每秒检查一次
        
        print(f"\n实时预测完成，共预测 {prediction_count} 次")

class MegaInformerConfig:
    """超大型Informer配置"""
    
    def __init__(self):
        # 模型参数
        self.seq_len = 30
        self.label_len = 0
        self.pred_len = 1
        
        # 特征参数
        self.enc_in = 33
        self.dec_in = 33
        self.c_out = 1
        
        # 超大型模型架构
        self.d_model = 256
        self.n_heads = 16
        self.e_layers = 4
        self.d_layers = 2
        self.d_ff = 1024
        self.factor = 5
        self.distil = True
        
        # 训练参数
        self.dropout = 0.1
        self.activation = 'gelu'
        self.embed = 'timeF'
        self.freq = 't'

def main():
    """主函数"""
    print("超大型比特币模型实时预测器")
    print("="*60)
    
    # 模型路径
    model_path = 'checkpoints/bitcoin_mega_classification/mega_best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先训练超大型模型")
        return
    
    # 创建预测器
    predictor = MegaRealtimeBitcoinPredictor(model_path)
    
    # 加载数据 - 使用缓存机制
    data_path = 'dataset/bitcoin/Bitcoin_BTCUSDT.csv'
    cache_path = 'mega_prediction_cache.pkl'
    
    # 检查缓存
    if os.path.exists(cache_path):
        print("正在从缓存加载数据...")
        import pickle
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        features = cached_data['features']
        print(f"✓ 从缓存加载完成，记录数: {len(features)}")
    else:
        print(f"正在加载数据: {data_path}")
        
        # 只加载最新的100万条数据用于预测
        print("为了加快加载速度，只加载最新的100万条数据...")
        df = pd.read_csv(data_path, nrows=1000000, skiprows=range(1, 3608278))  # 跳过前面的数据
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 创建特征
        print("正在创建增强特征...")
        features = predictor.create_features(df)
        
        # 保存缓存
        print("正在保存缓存...")
        import pickle
        with open(cache_path, 'wb') as f:
            pickle.dump({'features': features}, f)
        
        print(f"数据加载完成，总记录数: {len(features)}")
        print(f"特征维度: {features.shape}")
    
    # -----------------------------
    # 可选：只保留算子火焰图采集
    # -----------------------------
    run_operator_profiling = True       # 置为 True 采集算子火焰图（生成 TensorBoard 日志）

    if run_operator_profiling:
        # 采集一次火焰图（注意：会产生一定运行时间与日志体积）
        predictor.profile_inference(features, batch_size=64, steps=8, logdir='./tb_logs_realtime')

    # 如何查看火焰图TensorBoard: 在项目根目录运行 tensorboard --logdir ./tb_logs_realtime，浏览器访问后即可查看时间线/算子耗时。
    
    # 自动选择批量预测演示
    print("\n开始批量预测演示...")
    try:
        predictor.predict_batch(features, num_predictions=200)
    except KeyboardInterrupt:
        print("\n预测被用户中断")
    except Exception as e:
        print(f"预测过程中出现错误: {e}")

if __name__ == "__main__":
    main()
