#!/usr/bin/env python3
"""
超大型比特币分类模型训练器
目标：将参数量增加到百万级别
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import os
import json
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

from InformerClassification import create_informer_classification_model, InformerConfig
from bitcoin_optimized_features import BitcoinOptimizedFeatureEngineer

class MegaBitcoinClassificationTrainer:
    """超大型比特币分类模型训练器"""
    
    def __init__(self, config=None):
        self.config = config or MegaInformerConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        
        print(f"使用设备: {self.device}")
    
    def load_data(self, data_path, test_size=0.2):
        """加载和预处理数据"""
        print("正在加载增强特征数据...")
        
        # 读取完整数据
        print("正在读取完整数据集...")
        df = pd.read_csv(data_path)  # 使用完整数据
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"数据总量: {len(df)} 条记录")
        
        # 创建增强特征
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
        
        # 创建标签
        future_close = df['close'].shift(-1)
        current_close = df['close']
        labels = (future_close > current_close).astype(int)
        
        df = df[:-1]
        labels = labels[:-1]
        
        print(f"标签分布: 上涨={labels.sum()}, 下跌={len(labels)-labels.sum()}")
        print(f"上涨比例: {labels.mean():.3f}")
        
        # 选择特征
        X = df[feature_names].values
        y = labels.values
        
        # 标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 创建时间序列数据
        seq_len = self.config.seq_len
        X_sequences = []
        y_sequences = []
        
        for i in range(seq_len, len(X_scaled)):
            X_sequences.append(X_scaled[i-seq_len:i])
            y_sequences.append(y[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"创建的时间序列数量: {len(X_sequences)}")
        print(f"特征维度: {X_sequences.shape}")
        
        # 分割数据
        split_idx = int(len(X_sequences) * (1 - test_size))
        
        X_train = X_sequences[:split_idx]
        y_train = y_sequences[:split_idx]
        X_val = X_sequences[split_idx:]
        y_val = y_sequences[split_idx:]
        
        print(f"训练集大小: {len(X_train)}")
        print(f"验证集大小: {len(X_val)}")
        
        return (X_train, y_train), (X_val, y_val), scaler, feature_names
    
    def create_model(self, feature_dim):
        """创建超大型模型"""
        # 更新配置中的特征维度
        self.config.enc_in = feature_dim
        
        model, _ = create_informer_classification_model(self.config)
        model = model.to(self.device)
        
        self.model = model
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.lr * 0.01
        )
        
        # 使用加权损失函数处理数据不均衡
        class_weights = torch.tensor([1.0, 1.2], device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"超大型模型创建完成，特征维度: {feature_dim}")
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print(f"参数量级别: {total_params / 1_000_000:.1f}M")
        
        return model
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            target = target.squeeze()
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # 使用分类logits
            loss = self.criterion(output['class_logits'], target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output['class_logits'].argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                target = target.squeeze()
                
                output = self.model(data)
                loss = self.criterion(output['class_logits'], target)
                
                total_loss += loss.item()
                pred = output['class_logits'].argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(output['up_probability'].cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # 计算F1分数
        from sklearn.metrics import f1_score
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        # 计算AUC
        try:
            auc = roc_auc_score(all_targets, all_probs)
        except:
            auc = 0.5
        
        return avg_loss, accuracy, f1, auc
    
    def train(self, train_data, val_data, epochs=None):
        """训练模型"""
        epochs = epochs or self.config.epochs
        
        # 创建数据加载器
        train_loader = DataLoader(
            torch.utils.data.TensorDataset(
                torch.FloatTensor(train_data[0]),
                torch.LongTensor(train_data[1])
            ),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            torch.utils.data.TensorDataset(
                torch.FloatTensor(val_data[0]),
                torch.LongTensor(val_data[1])
            ),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        print(f"\n开始训练超大型模型，共 {epochs} 个epochs...")
        print("="*60)
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc, val_f1, val_auc = self.validate_epoch(val_loader)
            
            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.val_f1_scores.append(val_f1)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
            print(f"  验证F1: {val_f1:.4f}, 验证AUC: {val_auc:.4f}")
            print(f"  最佳验证准确率: {best_val_acc:.2f}%")
            print(f"  学习率: {current_lr:.6f}")
            print(f"  时间: {epoch_time:.1f}s")
            print("-" * 60)
        
        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n✓ 已加载最佳模型，验证准确率: {best_val_acc:.2f}%")
        
        return best_val_acc
    
    def save_model(self, save_dir, feature_names, scaler):
        """保存模型和相关信息"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(save_dir, 'mega_best_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'feature_names': feature_names,
            'scaler': scaler,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_f1_scores': self.val_f1_scores
        }, model_path)
        
        # 保存配置
        config_path = os.path.join(save_dir, 'mega_experiment_config.json')
        config_dict = self.config.__dict__.copy()
        config_dict['feature_count'] = len(feature_names)
        config_dict['feature_names'] = feature_names
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"超大型模型已保存到: {save_dir}")
        return model_path

class MegaInformerConfig:
    """超大型Informer配置 - 目标参数量百万级别"""
    
    def __init__(self):
        # 模型参数
        self.seq_len = 30
        self.label_len = 0
        self.pred_len = 1
        
        # 特征参数
        self.enc_in = 33  # 增强特征数量
        self.dec_in = 33
        self.c_out = 1
        
        # 超大型模型架构配置
        self.d_model = 256      # 大幅增加模型维度 (原来128 -> 256)
        self.n_heads = 16       # 增加注意力头数 (原来8 -> 16)
        self.e_layers = 4       # 增加编码器层数 (原来3 -> 4)
        self.d_layers = 2       # 增加解码器层数 (原来1 -> 2)
        self.d_ff = 1024        # 大幅增加前馈网络维度 (原来512 -> 1024)
        self.factor = 5
        self.distil = True
        
        # 训练参数
        self.dropout = 0.1
        self.activation = 'gelu'
        self.embed = 'timeF'
        self.freq = 't'
        
        # 优化器参数
        self.lr = 0.0002        # 降低学习率
        self.weight_decay = 1e-5
        self.batch_size = 8096   # 增加批次大小 (完整数据集)
        self.epochs = 1        # 减少训练轮数 (因为数据更多)

def main():
    """主训练函数"""
    print("超大型比特币分类模型训练 (目标：百万级参数)")
    print("="*70)
    
    # 创建训练器
    trainer = MegaBitcoinClassificationTrainer()
    
    # 加载数据
    data_path = 'dataset/bitcoin/Bitcoin_BTCUSDT.csv'
    train_data, val_data, scaler, feature_names = trainer.load_data(data_path)
    
    # 创建超大型模型
    feature_dim = len(feature_names)
    model = trainer.create_model(feature_dim)
    
    # 训练模型
    best_acc = trainer.train(train_data, val_data)
    
    # 保存模型
    save_dir = 'checkpoints/bitcoin_mega_classification'
    model_path = trainer.save_model(save_dir, feature_names, scaler)
    
    print(f"\n训练完成!")
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"超大型模型已保存到: {model_path}")

if __name__ == "__main__":
    main()
