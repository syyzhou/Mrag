#!/usr/bin/env python3
"""
查看 PyTorch .pth 文件内容的工具
使用方法: python view_pth.py <path_to_pth_file>
"""

import torch
import sys
import os
from pathlib import Path

def inspect_pth_file(filepath):
    """查看 .pth 文件内容"""
    
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return None
    
    print(f"\n{'='*60}")
    print(f"📁 文件: {filepath}")
    print(f"📏 大小: {os.path.getsize(filepath) / 1024:.2f} KB")
    print(f"{'='*60}\n")
    
    try:
        # 加载文件
        data = torch.load(filepath, map_location='cpu')
        
        # 检查类型
        if isinstance(data, dict):
            print(f"✅ 类型: 字典 (Dictionary)")
            print(f"📊 键数量: {len(data.keys())}")
            print(f"\n🔑 所有键: {list(data.keys())}\n")
            
            # 详细显示每个键的内容
            for key, value in data.items():
                print(f"{'─'*50}")
                print(f"📌 键: {key}")
                print(f"   类型: {type(value).__name__}")
                
                if isinstance(value, dict):
                    print(f"   子键数量: {len(value)}")
                    print(f"   子键: {list(value.keys())[:10]}")  # 最多显示10个
                    # 尝试显示部分内容
                    for sub_key, sub_val in list(value.items())[:3]:
                        if isinstance(sub_val, torch.Tensor):
                            print(f"     └─ {sub_key}: tensor shape {sub_val.shape}")
                        else:
                            print(f"     └─ {sub_key}: {sub_val}")
                
                elif isinstance(value, torch.Tensor):
                    print(f"   形状: {value.shape}")
                    print(f"   数据类型: {value.dtype}")
                    print(f"   数值范围: [{value.min().item():.4f}, {value.max().item():.4f}]")
                    print(f"   平均值: {value.mean().item():.4f}")
                    if value.numel() <= 10:
                        print(f"   数值: {value.tolist()}")
                    else:
                        print(f"   前5个值: {value.flatten()[:5].tolist()}")
                
                elif isinstance(value, (list, tuple)):
                    print(f"   长度: {len(value)}")
                    if len(value) > 0:
                        print(f"   第一个元素类型: {type(value[0]).__name__}")
                        if len(value) <= 10:
                            print(f"   内容: {value}")
                        else:
                            print(f"   前5个: {value[:5]}")
                
                elif isinstance(value, (int, float, str, bool)):
                    print(f"   值: {value}")
                
                else:
                    print(f"   值: {value}")
            
            print(f"\n{'─'*50}")
            return data
            
        elif isinstance(data, torch.nn.Module):
            print(f"✅ 类型: PyTorch 模型 (Module)")
            print(f"\n模型结构:\n{data}")
            return data
            
        else:
            print(f"✅ 类型: {type(data).__name__}")
            print(f"内容: {data}")
            return data
            
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return None

def save_as_json(data, output_path):
    """将配置保存为JSON格式"""
    import json
    import numpy as np
    
    def convert_to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        else:
            return obj
    
    try:
        serializable = convert_to_serializable(data)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"✅ 已保存为: {output_path}")
    except Exception as e:
        print(f"❌ 保存JSON失败: {e}")

def main():
    # 默认文件路径（你可以修改这里）
    default_path = "artifacts_dropout04_wd3e3_dst_time/config.pth"
    
    # 获取命令行参数
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = default_path
        print(f"📝 使用默认路径: {filepath}")
        print(f"💡 提示: 可以运行 'python {sys.argv[0]} <文件路径>' 指定其他文件\n")
    
    # 检查文件是否存在
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        print(f"\n使用方法:")
        print(f"  python {sys.argv[0]} /path/to/your/file.pth")
        sys.exit(1)
    
    # 查看文件内容
    data = inspect_pth_file(filepath)
    
    # 如果是字典，询问是否保存为JSON
    if data and isinstance(data, dict) and len(data) > 0:
        print(f"\n{'='*60}")
        response = input("💾 是否保存为JSON文件? (y/n): ").lower()
        if response == 'y':
            json_path = filepath.replace('.pth', '.json')
            save_as_json(data, json_path)
    
    print(f"\n{'='*60}")
    print("✅ 完成!")

if __name__ == "__main__":
    main()