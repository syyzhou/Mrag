# Written by Peibo Li
# Original code based on https://github.com/dvlab-research/LongLoRA?tab=readme-ov-file
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

import json
import os
import math
import pickle
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
import transformers
from model_time import MoraModel
from typing import Dict, Optional, Sequence
import sys
from transformers import BitsAndBytesConfig
import config
import datetime  # ✅ 新增
import re

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"


# ✅ 新增：时间解析函数（与训练代码一致）
def parse_target_time_from_text(text: str) -> Optional[Dict[str, int]]:
    """
    从 question 文本中解析目标时间
    目标时间是最后一个 "At YYYY-MM-DD HH:MM:SS" 格式的时间
    """
    pattern = r'At (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})'
    matches = re.findall(pattern, text)
    
    if matches:
        date_str, time_str = matches[-1]
        target_time_str = f"{date_str} {time_str}"
        dt = datetime.datetime.strptime(target_time_str, '%Y-%m-%d %H:%M:%S')
        
        return {
            'hour': dt.hour,
            'weekday': dt.weekday(),
            'is_weekend': 1 if dt.weekday() >= 5 else 0
        }
    return None

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size during inference')
    parser.add_argument('--base_model', type=str, default="./Qwen2-1.5B")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--seq_len', type=int, default=4096, help='context length during evaluation')
    parser.add_argument('--context_size', type=int, default=4096, help='context size during fine-tuning')
    parser.add_argument('--peft_model', type=str, default=None, help='')
    parser.add_argument('--flash_attn', type=bool, default=False, help='')
    parser.add_argument('--model_path', type=str, default='', help='your model path')
    parser.add_argument('--data_path', type=str, default="./test.bin", help='')
    parser.add_argument('--output_dir', type=str, default="./outputmodels/finetune-36/", help='')
    parser.add_argument('--dataset_name', type=str, default="nyc", help='')
    parser.add_argument('--test_file', type=str, default="test_qa_pairs_kqt_100.txt", help='')
    args = parser.parse_args()
    return args


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def get_as_batch(data, seq_length, batch_size, device='cpu', sliding_window=256):
    all_ix = list(range(0, len(data) - seq_length, sliding_window))
    all_ix.pop()

    for idx in range(0, len(all_ix), batch_size):
        ix = all_ix[idx:idx + batch_size]
        assert all([idx + seq_length + 1 <= len(data) for idx in ix])
        x = torch.stack([torch.from_numpy((data[i:i + seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y


def iceildiv(x, y):
    return (x + y - 1) // y


def evaluate_prediction_accuracy(prediction, ground_truth):
    pred_poi_pattern1 = r"POI id (\d+)."
    pred_poi_pattern2 = r"(\d+)."
    
    if "POI id" in prediction:
        predicted_poi = re.search(pred_poi_pattern1, prediction).group(1)
    elif "." in prediction:
        predicted_poi = prediction[:-1]
    else:
        predicted_poi = prediction
    actual_poi = re.search(pred_poi_pattern1, ground_truth).group(1)

    return int(predicted_poi == actual_poi)

def diagnose_time_router(model, tokenizer, test_lines, device, num_samples=50):
    """诊断 time-aware routing 是否有效"""
    print("\n" + "=" * 60)
    print("🔍 Time-Aware Router 诊断")
    print("=" * 60)
    
    model.eval()
    
    # 收集统计数据
    gate_weights = []  # router (gate) 的权重：决定 router1 vs router2
    router1_weights = []  # 时间感知路由器的专家权重
    router2_weights = []  # 非时间感知路由器的专家权重
    time_vectors = []
    time_labels = []  # 记录时间信息
    
    import datetime
    def parse_time(text):
        pattern = r'At (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})'
        matches = re.findall(pattern, text)
        if matches:
            date_str, time_str = matches[-1]
            dt = datetime.datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M:%S')
            return {'hour': dt.hour, 'weekday': dt.weekday(), 'is_weekend': 1 if dt.weekday() >= 5 else 0}
        return None
    
    for idx, line in enumerate(test_lines[:num_samples]):
        try:
            prompt1, gt = line.split("<answer>:")
            time_info = parse_time(prompt1)
            if time_info is None:
                continue
            
            # 设置时间向量
            if hasattr(model, 'time_encoder') and model.time_encoder is not None:
                time_info_tensor = {
                    'hour': torch.tensor([time_info['hour']], dtype=torch.long, device=device),
                    'weekday': torch.tensor([time_info['weekday']], dtype=torch.long, device=device),
                    'is_weekend': torch.tensor([time_info['is_weekend']], dtype=torch.long, device=device),
                }
                with torch.no_grad():
                    time_vec = model.time_encoder(time_info_tensor)
                    model.router_manager.set_time_vector(time_vec)
                    time_vectors.append(time_vec.cpu().numpy().flatten())
            
            # 前向传播
            inputs = tokenizer(prompt1[:1000], return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                _ = model(**inputs)
            
            # 收集路由权重（取第一层作为代表）
            rm = model.router_manager
            
            # token_routers: [router1_layer0, router2_layer0, router1_layer1, router2_layer1, ...]
            if len(rm.token_routers) >= 2:
                r1 = rm.token_routers[0]  # router1 (时间感知)
                r2 = rm.token_routers[1]  # router2 (非时间感知)
                
                if r1.routing_weight is not None:
                    w1 = r1.routing_weight.mean(dim=(0, 1)).cpu().numpy()
                    router1_weights.append(w1)
                if r2.routing_weight is not None:
                    w2 = r2.routing_weight.mean(dim=(0, 1)).cpu().numpy()
                    router2_weights.append(w2)
            
            # routers: [gate_layer0, gate_layer1, ...]
            if len(rm.routers) >= 1:
                gate = rm.routers[0]
                if gate.routing_weight is not None:
                    g = gate.routing_weight.mean(dim=0).cpu().numpy()
                    gate_weights.append(g)
            
            time_labels.append(time_info)
            rm.clear()
            
        except Exception as e:
            continue
    
    # ===== 分析 1: Gate Router =====
    print("\n📊 1. Gate Router 分析（决定 router1 vs router2 权重）")
    print("-" * 50)
    if gate_weights:
        gate_array = np.array(gate_weights)
        mean_gate = gate_array.mean(axis=0)
        std_gate = gate_array.std(axis=0)
        print(f"   gate 权重均值: {mean_gate.round(4)}")
        print(f"   gate 权重标准差: {std_gate.round(4)}")
        print(f"   → router1(时间感知) 平均权重: {mean_gate[0]:.4f}")
        print(f"   → router2(非时间感知) 平均权重: {mean_gate[1]:.4f}")
        
        if mean_gate[0] < 0.3:
            print("   ⚠️ 警告: router1 权重很低，时间信息可能被忽略!")
        elif abs(mean_gate[0] - mean_gate[1]) < 0.1:
            print("   ⚠️ 警告: 两个 router 权重接近，gate 没有明显偏好")
        else:
            print("   ✅ gate 有明显的路由偏好")
    else:
        print("   ❌ 未收集到 gate 权重")
    
    # ===== 分析 2: Router1 vs Router2 =====
    print("\n📊 2. Router1 vs Router2 对比")
    print("-" * 50)
    if router1_weights and router2_weights:
        r1_array = np.array(router1_weights)
        r2_array = np.array(router2_weights)
        
        # 计算差异
        diff = np.abs(r1_array - r2_array).mean()
        print(f"   Router1 专家权重均值: {r1_array.mean(axis=0).round(4)}")
        print(f"   Router2 专家权重均值: {r2_array.mean(axis=0).round(4)}")
        print(f"   两者平均差异: {diff:.4f}")
        
        if diff < 0.01:
            print("   ⚠️ 警告: router1 和 router2 输出几乎相同!")
            print("      → 时间信息没有影响路由决策")
        elif diff < 0.05:
            print("   ⚠️ 注意: 差异较小，时间影响有限")
        else:
            print("   ✅ router1 和 router2 有明显差异")
        
        # 检查 router1 是否随时间变化
        print("\n   Router1 在不同时间段的表现:")
        morning_weights = [r1_array[i] for i, t in enumerate(time_labels) if 6 <= t['hour'] < 12]
        evening_weights = [r1_array[i] for i, t in enumerate(time_labels) if 18 <= t['hour'] < 24]
        
        if morning_weights and evening_weights:
            morning_mean = np.array(morning_weights).mean(axis=0)
            evening_mean = np.array(evening_weights).mean(axis=0)
            time_diff = np.abs(morning_mean - evening_mean).mean()
            print(f"      早晨(6-12点) 专家权重: {morning_mean.round(4)}")
            print(f"      晚上(18-24点) 专家权重: {evening_mean.round(4)}")
            print(f"      早晚差异: {time_diff:.4f}")
            
            if time_diff < 0.02:
                print("      ⚠️ 早晚权重差异很小，时间感知可能无效!")
            else:
                print("      ✅ 不同时间段有不同的路由模式")
    else:
        print("   ❌ 未收集到 router 权重")
    
    # ===== 分析 3: Time Encoder =====
    print("\n📊 3. Time Encoder 分析")
    print("-" * 50)
    if time_vectors:
        tv_array = np.array(time_vectors)
        print(f"   Time vector 维度: {tv_array.shape[1]}")
        print(f"   Time vector 范数范围: [{np.linalg.norm(tv_array, axis=1).min():.4f}, {np.linalg.norm(tv_array, axis=1).max():.4f}]")
        
        # 检查不同时间的 time_vector 差异
        if len(time_vectors) >= 10:
            # 随机选两个不同时间的样本计算相似度
            from itertools import combinations
            sims = []
            for i, j in list(combinations(range(len(time_vectors)), 2))[:50]:
                v1, v2 = time_vectors[i], time_vectors[j]
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                sims.append(sim)
            
            mean_sim = np.mean(sims)
            std_sim = np.std(sims)
            print(f"   样本间余弦相似度: {mean_sim:.4f} ± {std_sim:.4f}")
            
            if mean_sim > 0.99:
                print("   ⚠️ 警告: time_encoder 输出几乎相同，没有学到时间差异!")
            elif mean_sim > 0.95:
                print("   ⚠️ 注意: 相似度较高，时间区分度有限")
            else:
                print("   ✅ time_encoder 对不同时间有区分")
    else:
        print("   ❌ 未收集到 time vector")
    
    # ===== 总结 =====
    print("\n" + "=" * 60)
    print("📋 诊断总结")
    print("=" * 60)
    
    issues = []
    if gate_weights and np.array(gate_weights).mean(axis=0)[0] < 0.3:
        issues.append("Gate 给 router1(时间感知) 的权重太低")
    if router1_weights and router2_weights:
        if np.abs(np.array(router1_weights) - np.array(router2_weights)).mean() < 0.02:
            issues.append("Router1 和 Router2 输出几乎相同")
    if time_vectors and np.mean([np.dot(time_vectors[0], v)/(np.linalg.norm(time_vectors[0])*np.linalg.norm(v)+1e-8) for v in time_vectors[1:10]]) > 0.98:
        issues.append("Time Encoder 对不同时间输出相似")
    
    if issues:
        print("发现以下问题:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. ⚠️ {issue}")
        print("\n建议:")
        print("   - 检查 time_encoder 是否有梯度流入")
        print("   - 增大 time_encoder 的学习率")
        print("   - 考虑给时间相关的 loss 加权")
    else:
        print("   ✅ 时间感知路由看起来正常工作")
    
    return {
        'gate_weights': gate_weights,
        'router1_weights': router1_weights,
        'router2_weights': router2_weights,
        'time_vectors': time_vectors,
        'time_labels': time_labels
    }
    
def main(args):
    device = "cuda:2"
    seed = 2
    torch.cuda.set_device(device)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model_path = args.model_path
    output_dir = args.output_dir
    print("data path", args.data_path)
    print("base model", model_path)
    print("peft model", output_dir)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=4096,
        padding_side="right",
        use_fast=False,
    )

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(model_path)

    context_size = args.context_size if args.context_size > 0 else args.seq_len
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(151646)

    if output_dir:
        model = MoraModel.from_pretrained(model, output_dir)
        peft_weights = torch.load(output_dir + '/' + 'adapter_model.safetensors')
        model.load_state_dict(peft_weights, strict=False)

    model.eval()
    model.to(device)

    # 特殊token检查
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    generation_config = transformers.GenerationConfig(
        max_new_tokens=30,
        min_new_tokens=None,
        do_sample=False,
        num_beams=5,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.176,
        num_return_sequences=5
    )

    data_path = f'./datasets/{args.dataset_name}/preprocessed/'
    with open(data_path + f"{args.test_file}", "r") as file:
        lines = file.readlines()
        
    correct_predictions_1 = 0
    correct_predictions_5 = 0
    correct_predictions_10 = 0
    correct_list = []

    for index, line in tqdm(enumerate(lines), desc="Processing lines", total=len(lines)):
        prompt1, gt = line.split("<answer>:")
        
        # ========== ✅ 新增：解析时间信息 ==========
        time_info = parse_target_time_from_text(prompt1)
        # ==========================================
        
        tmp1, tmp2 = prompt1.split('Which POI id will user ')
        time = tmp1[-24:]
        user_id = tmp2.split(' visit')[0]
        prompt = prompt1.replace('<question>:', '<question>:') + '<answer>:' + f'{time}, user {user_id} will visit POI id '

        if len(tokenizer.tokenize(prompt)) >= 4096:
            continue

        with torch.no_grad():
            # ========== ✅ 新增：设置时间信息 ==========
            if (time_info is not None and 
            hasattr(model, 'time_encoder') and 
            model.time_encoder is not None):
                time_info_tensor = {
                    'hour': torch.tensor([time_info['hour']], dtype=torch.long, device=device),
                    'weekday': torch.tensor([time_info['weekday']], dtype=torch.long, device=device),
                    'is_weekend': torch.tensor([time_info['is_weekend']], dtype=torch.long, device=device),
                }
                time_vector = model.time_encoder(time_info_tensor)
                model.router_manager.set_time_vector(time_vector)
            # ==========================================
            
            if model.task_encoder is not None:
                prefix_tensors = tokenizer(prompt, padding=True, return_tensors='pt', add_special_tokens=False).to(device)
                embedding = getattr(model.base_model, "embed_tokens")
                hidden_states = embedding(prefix_tensors["input_ids"])
                task_embed = model.task_encoder(hidden_states, prefix_tensors["attention_mask"])
                model.router_manager.set_task_weight(task_embed)

        prompt = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**prompt, generation_config=generation_config)
        
        # ========== ✅ 新增：清理时间信息 ==========
        if hasattr(model, 'router_manager') and model.router_manager is not None:
            model.router_manager.clear()
        # ==========================================

        gt = gt.replace('[', '').replace(']', '')
        i = 0
        seen_predictions = set()
        
        while i < 5:
            try:
                prediction = tokenizer.decode(outputs[:, prompt.input_ids.shape[1]:][i],
                                              skip_special_tokens=True)
                prediction = re.match(r'^\d+', prediction).group(0)
                i += 1
                
                if prediction in seen_predictions:
                    continue
                seen_predictions.add(prediction)
                tmp = evaluate_prediction_accuracy(prediction, gt)
                
                if tmp:
                    if i == 1:
                        correct_list.append(index)
                        correct_predictions_1 += tmp
                        correct_predictions_5 += tmp
                    elif i < 6:
                        correct_predictions_5 += tmp
                        break
                    else:
                        break
            except:
                continue

    print(f'ACC@1:{correct_predictions_1 / len(lines)}')
    print(f'ACC@5:{correct_predictions_5 / len(lines)}')
    print(f'ACC@10:{correct_predictions_10 / len(lines)}')
    print(f'correct_index:{correct_list}')
    
    print("\n运行 Time Router 诊断...")
    diagnose_results = diagnose_time_router(model, tokenizer, lines, device, num_samples=50)


if __name__ == "__main__":
    args = parse_config()
    main(args)