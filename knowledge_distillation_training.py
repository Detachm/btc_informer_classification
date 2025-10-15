#!/usr/bin/env python3
"""
知识蒸馏训练脚本
教师模型 → 学生模型 (Balanced Informer)
使用logits和特征蒸馏进行知识传递
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import time
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入项目模块
from InformerClassification import InformerClassification, InformerConfig, create_informer_classification_model
from bitcoin_optimized_features import BitcoinOptimizedFeatureEngineer

class StudentInformerConfig:
    """学生模型配置 - Balanced Informer"""
    
    def __init__(self):
        # 基础参数
        self.seq_len = 30
        self.label_len = 0
        self.pred_len = 1
        
        # 特征参数
        self.enc_in = 33  # 匹配教师模型的特征数
        self.dec_in = 33
        self.c_out = 1
        
        # 学生架构参数 (Balanced Informer)
        self.d_model = 96        # 嵌入维度 (教师: 128)
        self.n_heads = 2         # 注意力头数 (教师: 8)  
        self.e_layers = 2        # 编码层数 (教师: 4)
        self.d_layers = 1
        self.d_ff = 192          # 前馈维度 (教师: 512)
        self.factor = 5
        self.distil = True
        
        # 训练参数
        self.dropout = 0.1
        self.activation = 'gelu'
        self.embed = 'timeF'
        self.freq = 't'
        
        # 分类参数
        self.num_class = 2

class DistillationLoss(nn.Module):
    """知识蒸馏损失函数"""
    
    def __init__(self, temperature=4.0, alpha=0.7, beta=0.3):
        """
        Args:
            temperature: 蒸馏温度
            alpha: logits蒸馏权重
            beta: 特征蒸馏权重
        """
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        
    def forward(self, student_logits, teacher_logits, student_features, teacher_features, labels):
        """
        Args:
            student_logits: 学生模型logits [B, 2]
            teacher_logits: 教师模型logits [B, 2]
            student_features: 学生模型特征 [B, seq_len, d_model]
            teacher_features: 教师模型特征 [B, seq_len, d_model]
            labels: 真实标签 [B]
        """
        # 1. Logits蒸馏损失
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        kl_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # 2. 硬标签损失
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # 3. 特征蒸馏损失 (使用全局平均池化对齐维度)
        student_feat = student_features.mean(dim=1)  # [B, d_model]
        teacher_feat = teacher_features.mean(dim=1)  # [B, d_model]
        
        # 如果维度不匹配，使用线性投影对齐
        if student_feat.shape[1] != teacher_feat.shape[1]:
            if not hasattr(self, 'feature_proj'):
                self.feature_proj = nn.Linear(student_feat.shape[1], teacher_feat.shape[1]).to(student_feat.device)
            student_feat = self.feature_proj(student_feat)
        
        feature_loss = self.mse_loss(student_feat, teacher_feat)
        
        # 4. 组合损失
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * hard_loss + self.beta * feature_loss
        
        return {
            'total_loss': total_loss,
            'kl_loss': kl_loss,
            'hard_loss': hard_loss,
            'feature_loss': feature_loss
        }

class StudentInformerClassification(InformerClassification):
    """学生模型 - 基于Balanced Informer架构"""
    
    def __init__(self, configs):
        super(StudentInformerClassification, self).__init__(configs)
        
        # 重写编码器以使用学生架构参数
        from layers.Transformer_EncDec import Encoder, EncoderLayer, ConvLayer
        from layers.SelfAttention_Family import ProbAttention, AttentionLayer
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(configs.d_model) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        print(f"学生模型架构信息:")
        print(f"  d_model: {configs.d_model} (教师: 128)")
        print(f"  n_heads: {configs.n_heads} (教师: 8)")
        print(f"  e_layers: {configs.e_layers} (教师: 4)")
        print(f"  d_ff: {configs.d_ff} (教师: 512)")
    
    def forward_with_features(self, x_enc, x_mark_enc=None):
        """前向传播并返回中间特征"""
        # 编码
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # 保存中间特征用于蒸馏
        features = enc_out.clone()
        
        # 分类预测
        output = self.act(enc_out)
        output = self.dropout(output)
        
        # 展平
        output = output.reshape(output.shape[0], -1)
        
        # 动态创建投影层
        if self.projection is None:
            self.flatten_dim = output.shape[1]
            self.projection = nn.Linear(self.flatten_dim, self.num_class).to(output.device)
            self.prob_layer = nn.Sequential(
                nn.Linear(self.flatten_dim, 128),
                nn.ReLU(),
                nn.Dropout(self.dropout.p),
                nn.Linear(128, 1),
                nn.Sigmoid()
            ).to(output.device)
        
        # 二分类logits
        class_logits = self.projection(output)
        
        # 上涨概率
        up_prob = self.prob_layer(output)
        
        return {
            'class_logits': class_logits,
            'up_probability': up_prob.squeeze(-1),
            'features': features
        }

class KnowledgeDistillationTrainer:
    """知识蒸馏训练器"""
    
    def __init__(self, teacher_model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.teacher_model = None
        self.student_model = None
        self.distillation_loss = None
        self.optimizer = None
        self.scheduler = None
        
        # 训练历史
        self.train_losses = []
        self.val_accuracies = []
        self.distillation_metrics = []
        
        # 加载教师模型
        self.load_teacher_model(teacher_model_path)
        
        # 创建学生模型
        self.create_student_model()
        
        # 创建蒸馏损失函数
        self.distillation_loss = DistillationLoss(temperature=4.0, alpha=0.7, beta=0.3)
        
    def load_teacher_model(self, model_path):
        """加载预训练的教师模型"""
        print("正在加载教师模型...")
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取配置
        config_dict = checkpoint['config']
        teacher_config = InformerConfig()
        for key, value in config_dict.items():
            if hasattr(teacher_config, key):
                setattr(teacher_config, key, value)
        
        # 创建教师模型
        self.teacher_model, _ = create_informer_classification_model(teacher_config)
        self.teacher_model = self.teacher_model.to(self.device)
        
        # 处理动态创建的投影层
        model_state_dict = checkpoint['model_state_dict']
        
        # 触发投影层创建
        if 'projection.weight' in model_state_dict:
            dummy_input = torch.randn(1, teacher_config.seq_len, teacher_config.enc_in).to(self.device)
            with torch.no_grad():
                _ = self.teacher_model(dummy_input)
        
        # 加载状态字典
        self.teacher_model.load_state_dict(model_state_dict, strict=False)
        self.teacher_model.eval()
        
        # 计算参数量
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        print(f"教师模型参数量: {teacher_params:,} ({teacher_params/1_000_000:.1f}M)")
        print("✓ 教师模型加载完成")
        
        self.teacher_config = teacher_config
        
    def create_student_model(self):
        """创建学生模型"""
        print("正在创建学生模型...")
        
        # 创建学生配置
        student_config = StudentInformerConfig()
        
        # 确保特征维度匹配
        student_config.enc_in = self.teacher_config.enc_in
        
        # 创建学生模型
        self.student_model = StudentInformerClassification(student_config)
        self.student_model = self.student_model.to(self.device)
        
        # 计算参数量
        student_params = sum(p.numel() for p in self.student_model.parameters())
        compression_ratio = student_params / sum(p.numel() for p in self.teacher_model.parameters())
        
        print(f"学生模型参数量: {student_params:,} ({student_params/1_000_000:.1f}M)")
        print(f"压缩比: {compression_ratio:.3f}")
        print("✓ 学生模型创建完成")
        
        self.student_config = student_config
        
    def load_data(self, data_path, test_size=0.2, batch_size=1024):
        """加载训练数据"""
        print(f"正在加载数据: {data_path}")
        
        # 加载数据 - 只加载最新的数据用于训练
        print("为了加快加载速度，只加载最新的50万条数据...")
        df = pd.read_csv(data_path, nrows=500000, skiprows=range(1, 3108278))  # 跳过前面的数据
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 创建特征
        feature_engineer = BitcoinOptimizedFeatureEngineer()
        feature_names = feature_engineer.create_optimized_features(df)
        
        # 获取特征数据
        features = df[feature_names].values.astype(np.float32)
        
        # 检查并处理NaN值
        print(f"处理前 - NaN数量: {np.isnan(features).sum()}")
        print(f"处理前 - 无穷值数量: {np.isinf(features).sum()}")
        
        # 填充NaN值
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        print(f"处理后 - NaN数量: {np.isnan(features).sum()}")
        print(f"处理后 - 无穷值数量: {np.isinf(features).sum()}")
        print(f"特征值范围: [{features.min():.4f}, {features.max():.4f}]")
        
        # 创建标签 (价格变化方向)
        df['price_change'] = df['close'].pct_change()
        labels = (df['price_change'] > 0).astype(int).values
        
        # 对齐数据长度
        min_len = min(len(features), len(labels))
        features = features[:min_len]
        labels = labels[:min_len]
        
        # 创建序列数据
        seq_len = self.student_config.seq_len
        X, y = [], []
        
        for i in range(seq_len, len(features)):
            X.append(features[i-seq_len:i])
            y.append(labels[i])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"数据形状: X={X.shape}, y={y.shape}")
        
        # 划分训练集和验证集
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # 转换为张量
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.LongTensor(y_val)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        print(f"训练集: {len(train_dataset)} 样本")
        print(f"验证集: {len(val_dataset)} 样本")
        
        return train_loader, val_loader
        
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.student_model.train()
        total_loss = 0
        total_kl_loss = 0
        total_hard_loss = 0
        total_feature_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 教师模型前向传播 (不计算梯度)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(data)
                teacher_logits = teacher_outputs['class_logits']
                
                # 获取教师模型特征 (需要修改教师模型以返回特征)
                # 这里我们使用logits作为特征代理
                teacher_features = teacher_logits.unsqueeze(1).expand(-1, data.shape[1], -1)
            
            # 学生模型前向传播
            student_outputs = self.student_model.forward_with_features(data)
            student_logits = student_outputs['class_logits']
            student_features = student_outputs['features']
            
            # 计算蒸馏损失
            loss_dict = self.distillation_loss(
                student_logits, teacher_logits, 
                student_features, teacher_features, target
            )
            
            # 反向传播
            loss_dict['total_loss'].backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss_dict['total_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
            total_hard_loss += loss_dict['hard_loss'].item()
            total_feature_loss += loss_dict['feature_loss'].item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss_dict["total_loss"].item():.4f}, '
                      f'KL: {loss_dict["kl_loss"].item():.4f}, '
                      f'Hard: {loss_dict["hard_loss"].item():.4f}, '
                      f'Feature: {loss_dict["feature_loss"].item():.4f}')
        
        return {
            'total_loss': total_loss / len(train_loader),
            'kl_loss': total_kl_loss / len(train_loader),
            'hard_loss': total_hard_loss / len(train_loader),
            'feature_loss': total_feature_loss / len(train_loader)
        }
    
    def validate(self, val_loader):
        """验证模型"""
        self.student_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.student_model(data)
                
                # 使用概率阈值进行预测
                predictions = (outputs['up_probability'] > 0.5).long()
                total += target.size(0)
                correct += (predictions == target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        """执行知识蒸馏训练"""
        print(f"开始知识蒸馏训练，epochs={epochs}")
        
        # 创建优化器和调度器
        self.optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=lr * 0.01
        )
        
        best_acc = 0
        best_model_state = None
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_acc = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录指标
            self.train_losses.append(train_metrics['total_loss'])
            self.val_accuracies.append(val_acc)
            self.distillation_metrics.append(train_metrics)
            
            # 打印结果
            print(f"训练损失: {train_metrics['total_loss']:.4f}")
            print(f"  - KL损失: {train_metrics['kl_loss']:.4f}")
            print(f"  - 硬标签损失: {train_metrics['hard_loss']:.4f}")
            print(f"  - 特征损失: {train_metrics['feature_loss']:.4f}")
            print(f"验证准确率: {val_acc:.2f}%")
            print(f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = self.student_model.state_dict().copy()
                print(f"✓ 新的最佳验证准确率: {best_acc:.2f}%")
        
        # 加载最佳模型
        if best_model_state is not None:
            self.student_model.load_state_dict(best_model_state)
            print(f"\n✓ 训练完成，最佳验证准确率: {best_acc:.2f}%")
        
        return best_acc
    
    def save_model(self, save_dir, feature_names, scaler):
        """保存学生模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型检查点
        checkpoint = {
            'model_state_dict': self.student_model.state_dict(),
            'config': self.student_config.__dict__,
            'feature_names': feature_names,
            'scaler': scaler,
            'training_metrics': {
                'train_losses': self.train_losses,
                'val_accuracies': self.val_accuracies,
                'distillation_metrics': self.distillation_metrics
            },
            'timestamp': datetime.now().isoformat()
        }
        
        model_path = os.path.join(save_dir, 'student_best_model.pth')
        torch.save(checkpoint, model_path)
        
        # 保存训练报告
        report = {
            'student_architecture': self.student_config.__dict__,
            'training_summary': {
                'best_val_accuracy': max(self.val_accuracies) if self.val_accuracies else 0,
                'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
                'total_epochs': len(self.train_losses)
            },
            'compression_info': {
                'teacher_params': sum(p.numel() for p in self.teacher_model.parameters()),
                'student_params': sum(p.numel() for p in self.student_model.parameters()),
                'compression_ratio': sum(p.numel() for p in self.student_model.parameters()) / 
                                   sum(p.numel() for p in self.teacher_model.parameters())
            }
        }
        
        report_path = os.path.join(save_dir, 'student_training_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"学生模型已保存到: {model_path}")
        print(f"训练报告已保存到: {report_path}")
        
        return model_path

def main():
    """主函数"""
    print("知识蒸馏训练 - 教师模型 → 学生模型")
    print("="*60)
    
    # 检查教师模型
    teacher_model_path = 'checkpoints/bitcoin_mega_classification/mega_best_model.pth'
    if not os.path.exists(teacher_model_path):
        print(f"错误: 教师模型文件不存在: {teacher_model_path}")
        return
    
    # 创建蒸馏训练器
    trainer = KnowledgeDistillationTrainer(teacher_model_path)
    
    # 加载数据
    data_path = 'dataset/bitcoin/Bitcoin_BTCUSDT.csv'
    train_loader, val_loader = trainer.load_data(data_path, test_size=0.2, batch_size=512)
    
    # 执行蒸馏训练 (只训练1个epoch用于性能测试)
    best_acc = trainer.train(train_loader, val_loader, epochs=1, lr=0.0001)
    
    # 保存模型
    save_dir = 'checkpoints/student_distillation'
    
    # 从教师模型获取特征名称和标准化器
    teacher_checkpoint = torch.load(teacher_model_path, map_location='cpu')
    feature_names = teacher_checkpoint['feature_names']
    scaler = teacher_checkpoint['scaler']
    
    model_path = trainer.save_model(save_dir, feature_names, scaler)
    
    print(f"\n知识蒸馏训练完成!")
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"学生模型已保存到: {model_path}")

if __name__ == "__main__":
    main()
