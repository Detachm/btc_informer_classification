import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding


class InformerClassification(nn.Module):
    """
    基于Informer的二分类模型
    用于预测比特币价格方向（上涨/下跌）
    """

    def __init__(self, configs):
        super(InformerClassification, self).__init__()
        self.task_name = 'classification'
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.seq_len = configs.seq_len
        self.num_class = 2  # 二分类

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        
        # Encoder
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
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # 分类头
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        
        # 分类投影层 - 延迟初始化
        # 在第一次前向传播时动态创建
        self.projection = None
        self.prob_layer = None
        self.flatten_dim = None
        
        print(f"模型架构信息:")
        print(f"  seq_len: {configs.seq_len}")
        print(f"  d_model: {configs.d_model}")
        print(f"  enc_in: {configs.enc_in}")
        print(f"  e_layers: {configs.e_layers}")
        print("  投影层将在第一次前向传播时动态创建")

    def forward(self, x_enc, x_mark_enc=None):
        """
        Args:
            x_enc: 输入序列 [B, L, D]
            x_mark_enc: 时间标记（可选）
        """
        # 编码
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 分类预测
        output = self.act(enc_out)
        output = self.dropout(output)
        
        # 展平
        output = output.reshape(output.shape[0], -1)  # [B, seq_len * d_model]
        
        # 动态创建投影层（如果还没有创建）
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
            print(f"动态创建投影层，输入维度: {self.flatten_dim}")
        
        # 二分类logits
        class_logits = self.projection(output)  # [B, 2]
        
        # 上涨概率
        up_prob = self.prob_layer(output)  # [B, 1]
        
        return {
            'class_logits': class_logits,
            'up_probability': up_prob.squeeze(-1)  # [B]
        }

    def predict_proba(self, x_enc, x_mark_enc=None):
        """预测上涨概率"""
        with torch.no_grad():
            outputs = self.forward(x_enc, x_mark_enc)
            return outputs['up_probability'].cpu().numpy()

    def predict(self, x_enc, x_mark_enc=None, threshold=0.5):
        """预测类别"""
        with torch.no_grad():
            outputs = self.forward(x_enc, x_mark_enc)
            probs = outputs['up_probability']
            predictions = (probs > threshold).long()
            return predictions.cpu().numpy()


class InformerConfig:
    """Informer配置类"""
    
    def __init__(self):
        # 模型参数
        self.seq_len = 30  # 输入序列长度（匹配实际训练数据）
        self.label_len = 0  # 标签长度（分类任务不需要）
        self.pred_len = 1  # 预测长度
        
        # 特征参数
        self.enc_in = 15  # 输入特征数（匹配检查点）
        self.dec_in = 15
        self.c_out = 1
        
        # 模型架构
        self.d_model = 64
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 256
        self.factor = 5
        self.distil = True
        
        # 训练参数
        self.dropout = 0.1
        self.activation = 'gelu'
        self.embed = 'timeF'
        self.freq = 't'
        
        # 分类参数
        self.num_class = 2


def create_informer_classification_model(config=None):
    """创建Informer分类模型"""
    if config is None:
        config = InformerConfig()
    
    model = InformerClassification(config)
    return model, config


if __name__ == "__main__":
    # 测试模型
    config = InformerConfig()
    model = InformerClassification(config)
    
    # 测试输入
    batch_size = 4
    seq_len = config.seq_len
    enc_in = config.enc_in
    
    x_enc = torch.randn(batch_size, seq_len, enc_in)
    
    # 前向传播
    outputs = model(x_enc)
    
    print(f"输入形状: {x_enc.shape}")
    print(f"分类logits形状: {outputs['class_logits'].shape}")
    print(f"上涨概率形状: {outputs['up_probability'].shape}")
    
    # 预测
    probs = model.predict_proba(x_enc)
    predictions = model.predict(x_enc)
    
    print(f"上涨概率: {probs}")
    print(f"预测结果: {predictions}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
