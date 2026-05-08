#!/usr/bin/env python
"""
自动检测40GB显卡，执行训练和测试的脚本
当检测到有40GB显存的GPU时，执行训练脚本，然后自动执行测试
"""

import os
import sys
import torch
import subprocess
import argparse
from datetime import datetime


def check_gpu_memory():
    """检测GPU显存，返回有40GB显存的GPU设备"""
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return None
    
    print(f"✓ 检测到 {torch.cuda.device_count()} 个GPU")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"  GPU {i}: {props.name} - {memory_gb:.1f}GB")
        
        # 检查是否有40GB显存（允许误差范围：38-42GB）
        if 38 <= memory_gb <= 42:
            print(f"✓ GPU {i} 有约40GB显存 ({memory_gb:.1f}GB) - 将使用此GPU进行训练")
            return i
    
    return None


def run_training(gpu_id, train_data_path, output_dir, training_args):
    """执行训练命令"""
    print(f"\n{'='*80}")
    print(f"开始训练 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"GPU ID: {gpu_id}")
    print(f"训练数据: {train_data_path}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}\n")
    
    # 构建训练命令
    train_script = training_args.pop("train_script", "supervised-fine-tune-qlora.py")
    train_cmd = ["python", train_script]
    for key, value in training_args.items():
        if value is None:
            continue
        train_cmd.append(f"--{key}")
        train_cmd.append(str(value))
    
    # 设置CUDA设备
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        # 执行训练
        result = subprocess.run(train_cmd, env=env, check=True)
        print(f"\n✓ 训练成功完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"错误代码: {e.returncode}")
        return False


def run_evaluation(gpu_id, model_path, output_dir, test_file, eval_args):
    """执行测试命令"""
    print(f"\n{'='*80}")
    print(f"开始测试 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"GPU ID: {gpu_id}")
    print(f"模型路径: {model_path}")
    print(f"测试文件: {test_file}")
    print(f"{'='*80}\n")
    
    # 构建测试命令
    eval_cmd = [
        "python", "eval_next_poi_loss.py",
        "--batch_size", "1",
        "--base_model", "./Qwen2-1.5B",
        "--seq_len", "8192",
        "--context_size", "8192",
        "--peft_model", model_path,
        "--model_path", model_path,
        "--output_dir", output_dir,
        "--test_file", test_file,
        "--dataset_name", "test",
    ]
    
    # 添加可选的评估参数
    for key, value in eval_args.items():
        if value is not None:
            eval_cmd.append(f"--{key}")
            if isinstance(value, bool):
                if value:
                    eval_cmd[-1] = f"--{key}"
            else:
                eval_cmd.append(str(value))
    
    # 设置CUDA设备
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        # 执行测试
        result = subprocess.run(eval_cmd, env=env, check=True)
        print(f"\n✓ 测试成功完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 测试失败 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"错误代码: {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="自动检测40GB显卡并执行训练和测试")
    
    # 必需参数
    parser.add_argument("--train_script", type=str, default="supervised-fine-tune-qlora.py",
                        help="训练脚本路径")
    parser.add_argument("--model_name_or_path", type=str, default="./Qwen2.5-3B",
                        help="基础模型路径")
    parser.add_argument("--train_data", type=str, default="rag/multiview_enriched_qa/train_qa_multiview.json",
                        help="训练数据路径")
    parser.add_argument("--output_dir", type=str, default="./output/train_multiview",
                        help="模型输出目录")
    parser.add_argument("--test_file", type=str, default="test_qa_pairs_kqt_100.txt",
                        help="测试文件名")
    
    # 训练参数
    parser.add_argument("--bf16", type=str, default="True",
                        help="是否启用 bf16")
    parser.add_argument("--use_flash_attn", type=str, default="False",
                        help="是否启用 flash attention")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="单GPU批大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2,
                        help="单GPU评估批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="梯度累积步数")
    parser.add_argument("--evaluation_strategy", type=str, default="no",
                        help="评估策略")
    parser.add_argument("--save_strategy", type=str, default="no",
                        help="保存策略")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="保存步数")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="权重衰减")
    parser.add_argument("--warmup_steps", type=int, default=20,
                        help="预热步数")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup",
                        help="学习率调度器类型")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="日志记录间隔")
    parser.add_argument("--logging_dir", type=str, default="./logs",
                        help="日志目录")
    parser.add_argument("--deepspeed_config", type=str, default="ds_configs/stage2.json",
                        help="Deepspeed 配置文件")
    parser.add_argument("--model_max_length", type=int, default=8192,
                        help="最大模型长度")
    parser.add_argument("--tf32", type=str, default="True",
                        help="是否启用 tf32")
    
    # 评估参数
    parser.add_argument("--trajectory_embedding_path", type=str, default=None,
                        help="预编码轨迹向量路径（可选）")
    
    # 控制参数
    parser.add_argument("--skip_training", action="store_true",
                        help="跳过训练，只执行测试")
    parser.add_argument("--skip_eval", action="store_true",
                        help="跳过测试，只执行训练")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"自动训练和评估脚本")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # 检测GPU显存
    print("正在检测GPU显存...")
    gpu_id = check_gpu_memory()
    
    if gpu_id is None:
        print("\n❌ 没有找到40GB显存的GPU，无法执行训练")
        print("请检查您的GPU配置")
        return 1
    
    # 执行训练
    if not args.skip_training:
        training_args = {
            "train_script": args.train_script,
            "model_name_or_path": args.model_name_or_path,
            "bf16": args.bf16,
            "output_dir": args.output_dir,
            "use_flash_attn": args.use_flash_attn,
            "dataset": args.train_data,
            "low_rank_training": "True",
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "evaluation_strategy": args.evaluation_strategy,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "save_strategy": args.save_strategy,
            "save_steps": args.save_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "lr_scheduler_type": args.lr_scheduler_type,
            "logging_steps": args.logging_steps,
            "logging_dir": args.logging_dir,
            "deepspeed": args.deepspeed_config,
            "model_max_length": args.model_max_length,
            "tf32": args.tf32,
        }
        
        success = run_training(gpu_id, args.train_data, args.output_dir, training_args)
        
        if not success:
            print("\n❌ 训练失败，中止流程")
            return 1
    
    # 执行评估
    if not args.skip_eval:
        eval_args = {}
        if args.trajectory_embedding_path:
            eval_args["trajectory_embedding_path"] = args.trajectory_embedding_path
        
        success = run_evaluation(gpu_id, args.output_dir, args.output_dir, args.test_file, eval_args)
        
        if not success:
            print("\n❌ 评估失败")
            return 1
    
    print(f"\n{'='*80}")
    print(f"✓ 所有任务完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
