#!/usr/bin/env python3
"""
将YOLO模型导出为safetensors格式的脚本
safetensors是一种安全且高效的模型权重存储格式，只保存模型的权重而不包含模型结构
"""

import argparse
import os
import sys
from ultralytics import YOLO # type: ignore
from safetensors.torch import save_file,save_model



def export_to_safetensors(model_path, output_path=None):
    """
    将YOLO模型导出为safetensors格式
    
    Args:
        model_path (str): 输入的YOLO模型路径 (.pt文件)
        output_path (str, optional): 输出的safetensors文件路径
    
    Returns:
        str: 输出文件路径
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    
    
    # 如果没有指定输出路径，则根据输入路径生成
    if output_path is None:
        base_name = os.path.splitext(model_path)[0]
        output_path = f"{base_name}.safetensors"
    
    
    # 加载YOLO模型
    print(f"正在加载模型: {model_path}")
    model = YOLO(model_path)
    print(model.model,file=open("yolov10n.txt","a"))
    
    tensors = model.model.state_dict() # type: ignore
    
    for k, v in tensors.items():
        print(str(k), v.shape)
    
    # 保存为safetensors格式
    save_model(model.model, output_path) # type: ignore
    
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="将YOLO模型导出为safetensors格式")
    parser.add_argument("--model_path", help="YOLO .pt 模型文件路径")
    parser.add_argument("-o", "--output", help="输出的 .safetensors 文件路径")
    
    args = parser.parse_args()
    
    try:
        export_to_safetensors(args.model_path, args.output)
        print("导出完成!")
    except Exception as e:
        print(f"导出过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()