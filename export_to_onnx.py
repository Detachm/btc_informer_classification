#!/usr/bin/env python3
"""
导出 Informer 分类模型为 ONNX（固定 L、D，仅 batch 动态），用于后续 TensorRT 构建 FP16 引擎
"""

import os
import torch
import argparse

# 复用你项目里的模型与配置
from InformerClassification import create_informer_classification_model
from mega_realtime_prediction import MegaInformerConfig  # 提供 seq_len/enc_in 等配置
from knowledge_distillation_training import StudentInformerClassification, StudentInformerConfig

class UpProbWrapper(torch.nn.Module):
    """
    包一层只导出用得到的输出：'up_probability' => [B]
    你的模型 forward 返回 dict，需要包装成单一张量输出以简化引擎
    """
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        o = self.m(x)
        return o['up_probability']  # [B]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_out', type=str, default='informer_cls.fp32.onnx', help='导出 ONNX 路径')
    parser.add_argument('--seq_len', type=int, default=30, help='序列长度 L')
    parser.add_argument('--enc_in', type=int, default=33, help='特征维度 D')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset 版本')
    parser.add_argument('--model_type', type=str, default='teacher', choices=['teacher', 'student'], help='模型类型')
    parser.add_argument('--model_path', type=str, default='checkpoints/student_distillation/student_best_model.pth', help='学生模型路径')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model_type == 'student':
        # 加载学生模型
        print("加载学生模型...")
        checkpoint = torch.load(args.model_path, map_location=device)
        config_dict = checkpoint['config']
        
        config = StudentInformerConfig()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        model = StudentInformerClassification(config)
        model = model.to(device)
        
        # 处理动态投影层
        model_state_dict = checkpoint['model_state_dict']
        if 'projection.weight' in model_state_dict:
            dummy_input = torch.randn(1, config.seq_len, config.enc_in).to(device)
            with torch.no_grad():
                _ = model(dummy_input)
        
        model.load_state_dict(model_state_dict, strict=False)
        model.eval()
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"学生模型参数量: {total_params:,} ({total_params/1_000_000:.1f}M)")
        
    else:
        # 配置与建模（教师模型）
        config = MegaInformerConfig()
        config.seq_len = args.seq_len
        config.enc_in = args.enc_in

        model, _ = create_informer_classification_model(config)
        model = model.to(device).eval()

    # 触发一次前向，让动态创建的投影层等子模块完成初始化
    dummy = torch.randn(1, config.seq_len, config.enc_in, device=device)
    with torch.no_grad():
        _ = model(dummy)

    wrapper = UpProbWrapper(model).to(device).eval()

    # 导出 ONNX（固定 L、D，仅 batch 动态）
    # 添加ONNX导出配置以处理兼容性问题
    export_kwargs = {
        'input_names': ['input'],
        'output_names': ['up_probability'],
        'opset_version': args.opset,
        'do_constant_folding': True,
        'dynamic_axes': {'input': {0: 'batch'}, 'up_probability': {0: 'batch'}},
        'keep_initializers_as_inputs': False,  # 优化ONNX图结构
        'verbose': False,  # 减少输出信息
    }
    
    # 设置ONNX导出模式
    torch.onnx.is_in_onnx_export = lambda: True
    
    try:
        torch.onnx.export(wrapper, dummy, args.onnx_out, **export_kwargs)
        print(f'✓ 已导出 ONNX: {os.path.abspath(args.onnx_out)}')
    except Exception as e:
        print(f'❌ ONNX导出失败: {e}')
        print('尝试使用更兼容的导出设置...')
        
        # 使用更保守的导出设置
        export_kwargs.update({
            'opset_version': 11,  # 使用更稳定的opset版本
            'do_constant_folding': False,  # 禁用常量折叠
        })
        
        try:
            torch.onnx.export(wrapper, dummy, args.onnx_out, **export_kwargs)
            print(f'✓ 使用兼容模式导出成功: {os.path.abspath(args.onnx_out)}')
        except Exception as e2:
            print(f'❌ 兼容模式导出也失败: {e2}')
            raise
    finally:
        # 恢复ONNX导出模式
        torch.onnx.is_in_onnx_export = lambda: False

if __name__ == '__main__':
    main()