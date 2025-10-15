#!/usr/bin/env python3
"""
项目独立性验证脚本
检查项目是否完全独立，不依赖外部Time-Series-Library
"""

import sys
import os
import importlib

def test_imports():
    """测试所有核心模块导入"""
    print("🔍 测试项目独立性...")
    print("=" * 60)
    
    # 测试核心模型
    try:
        from InformerClassification import InformerClassification, InformerConfig, create_informer_classification_model
        print("✅ InformerClassification 导入成功")
    except Exception as e:
        print(f"❌ InformerClassification 导入失败: {e}")
        return False
    
    # 测试知识蒸馏
    try:
        from knowledge_distillation_training import StudentInformerClassification, StudentInformerConfig
        print("✅ knowledge_distillation_training 导入成功")
    except Exception as e:
        print(f"❌ knowledge_distillation_training 导入失败: {e}")
        return False
    
    # 测试特征工程
    try:
        from bitcoin_optimized_features import BitcoinOptimizedFeatureEngineer
        print("✅ bitcoin_optimized_features 导入成功")
    except Exception as e:
        print(f"❌ bitcoin_optimized_features 导入失败: {e}")
        return False
    
    # 测试layers模块
    try:
        from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
        from layers.SelfAttention_Family import ProbAttention, AttentionLayer
        from layers.Embed import DataEmbedding
        print("✅ layers 模块导入成功")
    except Exception as e:
        print(f"❌ layers 模块导入失败: {e}")
        return False
    
    # 测试utils模块
    try:
        from utils.masking import TriangularCausalMask, ProbMask
        from utils.metrics import RSE, CORR, MAE, MSE, RMSE, MAPE, MSPE
        print("✅ utils 模块导入成功")
    except Exception as e:
        print(f"❌ utils 模块导入失败: {e}")
        return False
    
    return True

def test_model_creation():
    """测试模型创建"""
    print("\n🏗️ 测试模型创建...")
    print("=" * 60)
    
    try:
        from InformerClassification import InformerConfig, create_informer_classification_model
        
        # 创建配置
        config = InformerConfig()
        config.seq_len = 30
        config.enc_in = 33
        config.d_model = 256
        config.e_layers = 4
        
        # 创建模型
        model, _ = create_informer_classification_model(config)
        print("✅ 原始模型创建成功")
        
        # 测试学生模型
        from knowledge_distillation_training import StudentInformerConfig, StudentInformerClassification
        
        student_config = StudentInformerConfig()
        student_config.seq_len = 30
        student_config.enc_in = 33
        student_config.d_model = 96
        student_config.e_layers = 2
        
        student_model = StudentInformerClassification(student_config)
        print("✅ 学生模型创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False

def test_inference():
    """测试推理功能"""
    print("\n🧪 测试推理功能...")
    print("=" * 60)
    
    try:
        import torch
        from InformerClassification import InformerConfig, create_informer_classification_model
        
        # 创建模型
        config = InformerConfig()
        config.seq_len = 30
        config.enc_in = 33
        config.d_model = 256
        config.e_layers = 4
        
        model, _ = create_informer_classification_model(config)
        model.eval()
        
        # 创建测试输入
        test_input = torch.randn(1, 30, 33)
        
        # 推理测试
        with torch.no_grad():
            output = model(test_input)
            up_prob = output['up_probability']
        
        print(f"✅ 推理测试成功，输出: {up_prob.item():.6f}")
        return True
        
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        return False

def check_external_dependencies():
    """检查是否还有外部依赖"""
    print("\n🔍 检查外部依赖...")
    print("=" * 60)
    
    # 检查Python路径
    current_dir = os.getcwd()
    time_series_lib_path = '/home/liuhan/Time-Series-Library'  # 这是检查目标，不是引用
    
    if time_series_lib_path in sys.path:
        print(f"⚠️ 发现外部路径: {time_series_lib_path}")
        return False
    else:
        print("✅ 未发现外部Time-Series-Library路径")
    
    # 检查文件中的硬编码路径
    python_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    external_refs = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if '/home/liuhan/Time-Series-Library' in content and 'verify_independence.py' not in file_path:
                    external_refs.append(file_path)
        except:
            continue
    
    if external_refs:
        print(f"⚠️ 发现外部引用文件: {external_refs}")
        return False
    else:
        print("✅ 未发现外部路径引用")
    
    return True

def main():
    """主函数"""
    print("🚀 Bitcoin Informer Classification - 项目独立性验证")
    print("=" * 80)
    
    # 测试导入
    if not test_imports():
        print("\n❌ 导入测试失败")
        return False
    
    # 测试模型创建
    if not test_model_creation():
        print("\n❌ 模型创建测试失败")
        return False
    
    # 测试推理
    if not test_inference():
        print("\n❌ 推理测试失败")
        return False
    
    # 检查外部依赖
    if not check_external_dependencies():
        print("\n❌ 外部依赖检查失败")
        return False
    
    print("\n" + "=" * 80)
    print("🎉 项目独立性验证通过！")
    print("=" * 80)
    print("""
✅ 验证结果:
• 所有核心模块导入成功
• 模型创建功能正常
• 推理功能正常
• 无外部依赖
• 项目完全独立

🚀 项目状态: 可以独立部署到GitHub
📦 依赖安装: pip install -r requirements.txt
🧪 功能测试: python test_pytorch_models.py
""")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
