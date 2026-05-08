# Written by Peibo Li
# Modified: 加入轨迹向量辅助路由（测试阶段）

import json
import os
import math
import pickle
import gc
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
import transformers
from model5 import MoraModel
from typing import Dict, Optional, Sequence
import sys
from transformers import BitsAndBytesConfig
import config
import re
import io

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--base_model', type=str, default="./Qwen2-1.5B")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--seq_len', type=int, default=8192)
    parser.add_argument('--context_size', type=int, default=8192)
    parser.add_argument('--peft_model', type=str, default=None)
    parser.add_argument('--flash_attn', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default="./test.bin")
    parser.add_argument('--output_dir', type=str, default="./outputmodels/finetune-36/")
    parser.add_argument('--dataset_name', type=str, default="nyc")
    parser.add_argument('--test_file', type=str, default="test_qa_pairs_kqt_100.txt")
    # 新增：测试集轨迹向量路径
    parser.add_argument('--trajectory_embedding_path', type=str, default=None,
                        help='预编码的测试集轨迹向量路径 (.pt 或 .npy)')
    parser.add_argument('--device', type=str, default=None,
                        help='cuda device, e.g. cuda:0')
    parser.add_argument('--generation_use_cache', action='store_true',
                        help='启用 KV cache；更快但更占显存')
    parser.add_argument('--low_memory', action='store_true',
                        help='启用 transformers 的 low_memory 生成模式')
    parser.add_argument('--empty_cache_per_step', action='store_true',
                        help='每条样本后清理 Python 和 CUDA cache，减轻显存碎片')
    parser.add_argument('--low_cpu_mem_usage', action='store_true',
                        help='模型加载时启用 low_cpu_mem_usage')
    args = parser.parse_args()
    return args


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def load_trajectory_embeddings(path):
    """加载预编码的轨迹向量"""
    if path is None or not os.path.exists(path):
        print(f"No trajectory embeddings found at {path}")
        return None

    print(f"Loading trajectory embeddings from {path}...")
    if path.endswith(".pt"):
        embed_data = torch.load(path, map_location="cpu")
        embeddings = embed_data["embeddings"]  # [N, hidden_dim]
        print(f"  Loaded {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
        print(f"  Pooling: {embed_data.get('pooling', 'unknown')}")
        print(f"  Model: {embed_data.get('model_name', 'unknown')}")
    elif path.endswith(".npy"):
        embeddings = torch.from_numpy(np.load(path))
        print(f"  Loaded {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
    else:
        print(f"  Unknown format: {path}")
        return None

    return embeddings


def set_trajectory_embedding_for_routers(model, traj_emb, device):
    """
    将单条轨迹向量分发到模型的所有router。
    traj_emb: (hidden_dim,) 单条向量，会扩展为 (1, hidden_dim)
    """
    if traj_emb is None:
        return

    if not hasattr(model, "router_manager"):
        return

    # 确保是 (1, hidden_dim) 形状
    if traj_emb.dim() == 1:
        traj_emb = traj_emb.unsqueeze(0)

    traj_emb = traj_emb.to(dtype=torch.bfloat16, device=device)

    for router in model.router_manager.token_routers:
        if hasattr(router, 'set_trajectory_embedding'):
            router.set_trajectory_embedding(traj_emb)


def clear_router_cache(model):
    """清理router缓存"""
    if hasattr(model, "router_manager") and hasattr(model.router_manager, "clear"):
        model.router_manager.clear()


def evaluate_prediction_accuracy(prediction, ground_truth):
    pred_poi_pattern1 = r"POI id (\d+)."
    if "POI id" in prediction:
        predicted_poi = re.search(pred_poi_pattern1, prediction).group(1)
    elif "." in prediction:
        predicted_poi = prediction[:-1]
    else:
        predicted_poi = prediction
    actual_poi = re.search(pred_poi_pattern1, ground_truth).group(1)
    return int(predicted_poi == actual_poi)


def main(args):
    device = args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 2
    if torch.cuda.is_available() and device.startswith("cuda:"):
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
        model_max_length=8192,
        padding_side="right",
        use_fast=False,
    )

    model_config = transformers.AutoConfig.from_pretrained(model_path)

    context_size = args.context_size if args.context_size > 0 else args.seq_len
    orig_ctx_len = getattr(model_config, "max_position_embeddings", None)
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        model_config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )

    if output_dir:
        model = MoraModel.from_pretrained(model, output_dir)
        peft_weights = torch.load(output_dir + '/' + 'adapter_model.safetensors')
        model.load_state_dict(peft_weights, strict=False)

    model.eval()
    model.to(device)

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

    # ===== 加载测试集轨迹向量 =====
    traj_embeddings = load_trajectory_embeddings(args.trajectory_embedding_path)
    use_traj_routing = traj_embeddings is not None and hasattr(model, "router_manager")
    if use_traj_routing:
        print(f"Trajectory routing enabled: {traj_embeddings.shape[0]} embeddings loaded")
    else:
        print("Trajectory routing disabled (no embeddings or no router_manager)")

    generation_config = transformers.GenerationConfig(
        max_new_tokens=30,
        min_new_tokens=None,
        do_sample=False,
        num_beams=5,
        use_cache=args.generation_use_cache,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.176,
        num_return_sequences=5
    )

    # data_path = f'./datasets/{args.dataset_name}/preprocessed/'
    # data_path = f'./rag/oracle_sem_enriched_qa/'
    # data_path = f'./rag/sem_enriched_qa/'
    data_path = f''
    with open(data_path + f"{args.test_file}", "r") as file:
        lines = file.readlines()

    # 验证轨迹向量数量与测试样本数量匹配
    if use_traj_routing:
        if len(traj_embeddings) != len(lines):
            print(f"WARNING: embedding count ({len(traj_embeddings)}) != "
                  f"test sample count ({len(lines)}). "
                  f"Will use min({len(traj_embeddings)}, {len(lines)})")

    correct_predictions_1 = 0
    correct_predictions_5 = 0
    correct_predictions_10 = 0
    model.eval()

    correct_list = []
    skipped = 0

    for index, line in tqdm(enumerate(lines), desc="Processing lines", total=len(lines)):
        prompt1, gt = line.split("<answer>:")
        tmp1, tmp2 = prompt1.split('Which POI id will user ')
        time = tmp1[-24:]
        user_id = tmp2.split(' visit')[0]
        prompt = (prompt1.replace('<question>:', '<question>:') +
                  '<answer>:' + f'{time}, user {user_id} will visit POI id ')

        if len(tokenizer.tokenize(prompt)) >= 8192:
            print(len(tokenizer.tokenize(prompt)))
            skipped += 1
            continue

        with torch.no_grad():
            if args.empty_cache_per_step and device.startswith("cuda"):
                gc.collect()
                torch.cuda.empty_cache()

            # ===== 设置轨迹向量到router =====
            if use_traj_routing and index < len(traj_embeddings):
                set_trajectory_embedding_for_routers(
                    model, traj_embeddings[index], device)

            # task_encoder (如果有的话)
            if hasattr(model, 'task_encoder') and model.task_encoder is not None:
                prefix_tensors = tokenizer(
                    prompt,
                    padding=True,
                    return_tensors='pt',
                    add_special_tokens=False,
                    truncation=True,
                    max_length=args.context_size,
                ).to(device)
                embedding = getattr(model.base_model, "embed_tokens")
                hidden_states = embedding(prefix_tensors["input_ids"])
                task_embed = model.task_encoder(
                    hidden_states, prefix_tensors["attention_mask"])
                model.router_manager.set_task_weight(task_embed)

            prompt_tokens = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.context_size,
            ).to(device)
            print(index, prompt_tokens.input_ids.shape[1])
            generate_kwargs = dict(
                **prompt_tokens,
                generation_config=generation_config,
            )
            if args.low_memory:
                generate_kwargs["low_memory"] = True
            outputs = model.generate(**generate_kwargs)

            # ===== 清理router缓存 =====
            clear_router_cache(model)

        gt = gt.replace('[', '').replace(']', '')
        seen_predictions = set()
        max_candidates = min(5, outputs.shape[0])
        for i in range(max_candidates):
            try:
                prediction = tokenizer.decode(
                    outputs[:, prompt_tokens.input_ids.shape[1]:][i],
                    skip_special_tokens=True)
                prediction = re.match(r'^\d+', prediction).group(0)
                if prediction in seen_predictions:
                    continue
                seen_predictions.add(prediction)
                tmp = evaluate_prediction_accuracy(prediction, gt)
                if tmp:
                    rank = i + 1
                    if rank == 1:
                        correct_list.append(index)
                        correct_predictions_1 += tmp
                        correct_predictions_5 += tmp
                    elif rank < 6:
                        correct_predictions_5 += tmp
                        break
                    else:
                        break
            except:
                continue

        if args.empty_cache_per_step and device.startswith("cuda"):
            gc.collect()
            torch.cuda.empty_cache()

    total = len(lines)
    print(f'\nResults:')
    print(f'  Total samples: {total}, Skipped: {skipped}')
    print(f'  ACC@1: {correct_predictions_1 / total:.4f} ({correct_predictions_1}/{total})')
    print(f'  ACC@5: {correct_predictions_5 / total:.4f} ({correct_predictions_5}/{total})')
    print(f'  ACC@10: {correct_predictions_10 / total:.4f} ({correct_predictions_10}/{total})')
    if use_traj_routing:
        print(f'  (with trajectory routing)')
    print(f'  correct_index: {correct_list}')


if __name__ == "__main__":
    args = parse_config()
    main(args)
