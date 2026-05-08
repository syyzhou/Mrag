# Written by Peibo Li
# Modified: adapt test-time trajectory routing to current model8 implementation

import os
import math
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
import transformers
from model8 import MoraModel
from typing import Dict
import re

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--base_model', type=str, default="./Qwen2-1.5B")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--seq_len', type=int, default=4096)
    parser.add_argument('--context_size', type=int, default=4096)
    parser.add_argument('--peft_model', type=str, default=None)
    parser.add_argument('--flash_attn', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default="./test.bin")
    parser.add_argument('--output_dir', type=str, default="./outputmodels/finetune-36/")
    parser.add_argument('--dataset_name', type=str, default="nyc")
    parser.add_argument('--test_file', type=str, default="test_qa_pairs_kqt_100.txt")
    parser.add_argument('--test_path', type=str, default=None,
                        help='Direct path to test txt file. If set, overrides dataset_name/test_file.')
    parser.add_argument('--trajectory_embedding_path', type=str, default=None,
                        help='预编码的测试集轨迹向量路径 (.pt 或 .npy)')
    parser.add_argument('--device', type=str, default=None,
                        help='cuda device, e.g. cuda:0')
    return parser.parse_args()


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


def prepare_tokenizer_for_llama2(
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    if tokenizer.eos_token is None:
        tokenizer.eos_token = DEFAULT_EOS_TOKEN
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id


def load_trajectory_embeddings(path):
    if path is None or not os.path.exists(path):
        print(f"No trajectory embeddings found at {path}")
        return None

    print(f"Loading trajectory embeddings from {path}...")
    if path.endswith(".pt"):
        embed_data = torch.load(path, map_location="cpu")
        embeddings = embed_data["embeddings"]
        print(f"  Loaded {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
        print(f"  Pooling: {embed_data.get('pooling', 'unknown')}")
        print(f"  Model: {embed_data.get('model_name', 'unknown')}")
        stats = embed_data.get("extraction_stats", {})
        if stats:
            print("  Extraction stats:")
            print(f"    Original count: {stats.get('original_count', '?')}")
            print(f"    Matched count:  {stats.get('matched_count', '?')}")
            print(f"    Unmatched count: {stats.get('unmatched_count', '?')}")
            print(f"    Match rate: {stats.get('match_rate', '?'):.2f}%")
    elif path.endswith(".npy"):
        embeddings = torch.from_numpy(np.load(path))
        print(f"  Loaded {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
    else:
        print(f"  Unknown format: {path}")
        return None

    return embeddings


def _collect_traj_projectors(model):
    if not hasattr(model, "router_manager"):
        return []

    projectors = []
    seen = set()

    if hasattr(model, "traj_projector") and model.traj_projector is not None:
        seen.add(id(model.traj_projector))
        projectors.append(model.traj_projector)

    for router in model.router_manager.token_routers:
        if getattr(router, 'use_trajectory', False) and getattr(router, 'traj_projector', None) is not None:
            pid = id(router.traj_projector)
            if pid not in seen:
                seen.add(pid)
                projectors.append(router.traj_projector)

    return projectors


def set_trajectory_embedding_for_all_routers(model, traj_emb, device):
    """
    当前实现下，轨迹向量缓存于 TrajectoryProjector，而不是 TokenRouter/FusionGate 本身。
    """
    if traj_emb is None:
        return 0

    projectors = _collect_traj_projectors(model)
    if len(projectors) == 0:
        return 0

    if traj_emb.dim() == 1:
        traj_emb = traj_emb.unsqueeze(0)
    traj_emb = traj_emb.to(dtype=torch.bfloat16, device=device)

    for proj in projectors:
        proj.set_trajectory_embedding(traj_emb)

    return len(projectors)


def clear_all_router_cache(model, clear_traj_embedding=False):
    if hasattr(model, "router_manager") and hasattr(model.router_manager, "clear"):
        model.router_manager.clear()

    if clear_traj_embedding:
        for proj in _collect_traj_projectors(model):
            proj.clear()


def verify_router_trajectory_setup(model):
    if not hasattr(model, "router_manager"):
        print("  [verify] No router_manager found")
        return

    print("\n  [verify] Router trajectory setup:")
    miss = 0
    for i, router in enumerate(model.router_manager.token_routers):
        needs_traj = getattr(router, 'use_trajectory', False)
        if not needs_traj:
            continue
        proj = getattr(router, 'traj_projector', None)
        cached = getattr(proj, '_cached_traj_embedding', None) if proj is not None else None
        has_embedding = cached is not None
        status = "OK" if has_embedding else "MISSING!"
        if not has_embedding:
            miss += 1
        print(f"    TokenRouter[{i}] tag={getattr(router, 'tag', None)}: "
              f"use_trajectory={needs_traj}, has_embedding={has_embedding} [{status}]")

    if miss == 0:
        print("    All trajectory-enabled TokenRouter are ready.")
    print()


def count_routers(model):
    if not hasattr(model, "router_manager"):
        return {}

    stats = {
        'token_routers_total': len(model.router_manager.token_routers),
        'token_routers_with_traj': 0,
        'fusion_gates_total': len(model.router_manager.routers),
        'traj_projectors_total': len(_collect_traj_projectors(model)),
    }

    for router in model.router_manager.token_routers:
        if getattr(router, 'use_trajectory', False):
            stats['token_routers_with_traj'] += 1

    return stats


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
        model_max_length=4096,
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
    )

    if output_dir:
        model = MoraModel.from_pretrained(model, output_dir)
        peft_weights = torch.load(output_dir + '/' + 'adapter_model.safetensors', map_location='cpu')
        model.load_state_dict(peft_weights, strict=False)

    model.eval()
    model.to(device)

    prepare_tokenizer_for_llama2(tokenizer=tokenizer, model=model)

    traj_embeddings = load_trajectory_embeddings(args.trajectory_embedding_path)
    use_traj_routing = traj_embeddings is not None and hasattr(model, "router_manager")

    if use_traj_routing:
        print(f"\nTrajectory routing ENABLED:")
        print(f"  Embeddings: {traj_embeddings.shape[0]} samples, dim={traj_embeddings.shape[1]}")

        router_stats = count_routers(model)
        print(f"  TokenRouters: {router_stats['token_routers_total']} total, "
              f"{router_stats['token_routers_with_traj']} use trajectory")
        print(f"  FusionGates:  {router_stats['fusion_gates_total']} total")
        print(f"  TrajProjectors: {router_stats['traj_projectors_total']}")

        print("\n  Verifying with first sample...")
        set_trajectory_embedding_for_all_routers(model, traj_embeddings[0], device)
        verify_router_trajectory_setup(model)
        clear_all_router_cache(model, clear_traj_embedding=True)
    else:
        print("\nTrajectory routing DISABLED")
        if traj_embeddings is None:
            print("  Reason: no trajectory embeddings loaded")
        if not hasattr(model, "router_manager"):
            print("  Reason: model has no router_manager")

    generation_config = transformers.GenerationConfig(
        max_new_tokens=30,
        min_new_tokens=None,
        do_sample=False,
        num_beams=5,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.176,
        num_return_sequences=5
    )

    if args.test_path is not None:
        test_path = args.test_path
    else:
        test_path = f'./datasets/{args.dataset_name}/preprocessed/{args.test_file}'

    print("test path", test_path)
    with open(test_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    if use_traj_routing:
        n_embed = len(traj_embeddings)
        n_test = len(lines)
        if n_embed != n_test:
            print(f"\n  WARNING: embedding count ({n_embed}) != test sample count ({n_test})")
            print(f"  Will process min({n_embed}, {n_test}) = {min(n_embed, n_test)} samples with trajectory routing")
            print(f"  Remaining {max(0, n_test - n_embed)} samples will use uniform routing fallback")

    correct_predictions_1 = 0
    correct_predictions_5 = 0
    correct_predictions_10 = 0
    model.eval()

    correct_list = []
    skipped = 0
    traj_used_count = 0
    traj_fallback_count = 0

    for index, line in tqdm(enumerate(lines), desc="Processing lines", total=len(lines)):
        prompt1, gt = line.split("<answer>:")
        tmp1, tmp2 = prompt1.split('Which POI id will user ')
        time = tmp1[-24:]
        user_id = tmp2.split(' visit')[0]
        prompt = (prompt1.replace('<question>:', '<question>:') +
                  '<answer>:' + f'{time}, user {user_id} will visit POI id ')

        if len(tokenizer.tokenize(prompt)) >= 4096:
            skipped += 1
            continue

        with torch.no_grad():
            if use_traj_routing and index < len(traj_embeddings):
                set_trajectory_embedding_for_all_routers(model, traj_embeddings[index], device)
                traj_used_count += 1
            else:
                # 没有轨迹时显式清空 projector 缓存，避免沿用上一个样本。
                clear_all_router_cache(model, clear_traj_embedding=True)
                traj_fallback_count += 1

            if hasattr(model, 'task_encoder') and model.task_encoder is not None:
                prefix_tensors = tokenizer(
                    prompt, padding=True, return_tensors='pt',
                    add_special_tokens=False).to(device)
                embedding = getattr(model.base_model, "embed_tokens")
                hidden_states = embedding(prefix_tensors["input_ids"])
                task_embed = model.task_encoder(hidden_states, prefix_tensors["attention_mask"])
                model.router_manager.set_task_weight(task_embed)

            prompt_tokens = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(**prompt_tokens,
                                     generation_config=generation_config)

            clear_all_router_cache(model, clear_traj_embedding=True)

        gt = gt.replace('[', '').replace(']', '')
        i = 0
        seen_predictions = set()
        while i < 5:
            try:
                prediction = tokenizer.decode(
                    outputs[:, prompt_tokens.input_ids.shape[1]:][i],
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
            except Exception:
                continue

    total = len(lines)
    print(f'\n{"="*60}')
    print('Results:')
    print(f'{"="*60}')
    print(f'  Total samples: {total}')
    print(f'  Skipped (too long): {skipped}')
    print(f'  Evaluated: {total - skipped}')
    print(f'  ACC@1:  {correct_predictions_1 / total:.4f} ({correct_predictions_1}/{total})')
    print(f'  ACC@5:  {correct_predictions_5 / total:.4f} ({correct_predictions_5}/{total})')
    print(f'  ACC@10: {correct_predictions_10 / total:.4f} ({correct_predictions_10}/{total})')

    if use_traj_routing:
        print(f'\n  Trajectory routing stats:')
        print(f'    With trajectory:    {traj_used_count}')
        print(f'    Fallback (uniform): {traj_fallback_count}')

    print(f'\n  Correct@1 indices: {correct_list}')
    print(f'{"="*60}')


if __name__ == "__main__":
    args = parse_config()
    main(args)
