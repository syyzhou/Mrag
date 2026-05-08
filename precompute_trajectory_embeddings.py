# precompute_trajectory_embeddings.py
"""
用LLM对训练数据中的轨迹进行预编码，保存每条样本的向量表示。
"""

import os
import io
import json
import re
import torch
import numpy as np
from tqdm import tqdm
import transformers
from dataclasses import dataclass, field
from typing import Optional


def jload(f, mode="r"):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    jdict = json.load(f)
    f.close()
    return jdict


@dataclass
class Args:
    model_name_or_path: str = field(default="Qwen2.5-3B")
    dataset_name: str = field(default_factory=lambda: os.getenv("DATASET_NAME", "nyc"))
    dataset: Optional[str] = field(default=None)
    output_path: Optional[str] = field(default=None)
    split: str = field(default="train", metadata={"help": "train / test"})
    model_max_length: int = field(default=4096)
    batch_size: int = field(default=1)
    pooling: str = field(default="last", metadata={"help": "last / mean / weighted_mean"})
    cache_dir: Optional[str] = field(default=None)


def load_trajectory_txt(file_path):
    """
    读取 txt 格式的轨迹数据文件，返回 [{'question': ..., 'answer': ...}, ...] 列表
    每条记录以 <question>: ... <answer>: ... 分隔
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    pattern = r"<question>:\s*(.*?)\s*<answer>:\s*(.*?)\s*(?=(<question>:|$))"
    matches = re.findall(pattern, content, re.DOTALL)

    for q, a, _ in matches:
        data.append({
            "question": q.strip(),
            "answer": a.strip()
        })
    return data


def extract_trajectory_text(question: str, sample_index: int = -1) -> tuple:
    """
    从question中提取轨迹部分（check-in序列），去掉问题指令部分。
    这样编码的向量更聚焦于轨迹本身的语义。

    Returns:
        (trajectory_text, matched): 
            trajectory_text: 提取到的轨迹文本
            matched: 是否成功匹配到轨迹部分 (True/False)
    """
    match = re.search(
        r'(The following data contains.*?)(?:Given the data,)',
        question,
        re.DOTALL
    )
    if match:
        return match.group(1).strip(), True
    else:
        # 未匹配，返回整个question并标记
        return question, False


def compute_embeddings(model, tokenizer, texts, args, device):
    """
    批量计算文本的嵌入向量
    """
    all_embeddings = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch_size), desc="Encoding trajectories"):
            batch_texts = texts[i:i + args.batch_size]

            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.model_max_length,
            ).to(device)

            outputs = model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
                output_hidden_states=True,
            )

            # 取最后一层hidden states
            hidden_states = outputs.hidden_states[-1]  # [B, seq_len, hidden_dim]
            attention_mask = encodings["attention_mask"]  # [B, seq_len]

            if args.pooling == "last":
                seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
                embeddings = hidden_states[
                    torch.arange(hidden_states.size(0), device=device),
                    seq_lengths
                ]  # [B, hidden_dim]

            elif args.pooling == "mean":
                mask_expanded = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
                sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                count = mask_expanded.sum(dim=1).clamp(min=1)
                embeddings = sum_hidden / count

            elif args.pooling == "weighted_mean":
                positions = torch.arange(hidden_states.size(1), device=device).float()
                weights = positions.unsqueeze(0) * attention_mask.float()
                weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1)
                embeddings = (hidden_states * weights.unsqueeze(-1)).sum(dim=1)
            else:
                raise ValueError(f"Unknown pooling: {args.pooling}")

            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)  # [N, hidden_dim]


def main():
    parser = transformers.HfArgumentParser(Args)
    args = parser.parse_args_into_dataclasses()[0]
    if args.split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {args.split}. Expected one of: train, test")

    data_dir = f"./datasets/{args.dataset_name}/preprocessed"
    if args.dataset is None:
        args.dataset = (
            f"{data_dir}/train_qa_pairs_kqt.json"
            if args.split == "train"
            else f"{data_dir}/test_qa_pairs_kqt.txt"
        )
    if args.output_path is None:
        args.output_path = (
            f"{data_dir}/trajectory_embeddings.pt"
            if args.split == "train"
            else f"{data_dir}/test_embeddings.pt"
        )

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.model_name_or_path}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="left",
        use_fast=False,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
    ).to(device)
    model.eval()

    print(f"Loading dataset from {args.dataset}...")
    if args.dataset.endswith(".json"):
        data = jload(args.dataset)
    elif args.dataset.endswith(".txt"):
        data = load_trajectory_txt(args.dataset)
    else:
        raise ValueError(f"Unsupported dataset format: {args.dataset}")

    original_count = len(data)
    print(f"Original dataset size: {original_count}")

    # ---- 提取轨迹文本，并统计匹配情况 ----
    trajectory_texts = []
    matched_count = 0
    unmatched_count = 0
    unmatched_indices = []

    for idx, item in enumerate(data):
        traj_text, matched = extract_trajectory_text(item["question"], sample_index=idx)
        trajectory_texts.append(traj_text)

        if matched:
            matched_count += 1
        else:
            unmatched_count += 1
            unmatched_indices.append(idx)

    # ---- 打印匹配统计 ----
    print("\n" + "=" * 60)
    print("Trajectory Extraction Statistics")
    print("=" * 60)
    print(f"  Original samples:      {original_count}")
    print(f"  Successfully matched:  {matched_count}")
    print(f"  Unmatched (fallback):  {unmatched_count}")
    print(f"  Match rate:            {matched_count / original_count * 100:.2f}%")
    print(f"  Output trajectory count: {len(trajectory_texts)}")
    print(f"  Count consistent:      {'YES' if len(trajectory_texts) == original_count else 'NO !!!'}")

    if unmatched_count > 0:
        print(f"\n  Unmatched sample indices (first 20): {unmatched_indices[:20]}")
        if unmatched_count <= 5:
            # 少量未匹配时，打印内容帮助调试
            for ui in unmatched_indices[:5]:
                q_preview = data[ui]["question"][:200].replace("\n", " ")
                print(f"    [idx={ui}] question preview: {q_preview}...")
        else:
            # 只打印第一个
            ui = unmatched_indices[0]
            q_preview = data[ui]["question"][:300].replace("\n", " ")
            print(f"    [idx={ui}] first unmatched question preview:")
            print(f"      {q_preview}...")

    # 严格校验数量一致性
    assert len(trajectory_texts) == original_count, \
        (f"FATAL: trajectory count ({len(trajectory_texts)}) != "
         f"original count ({original_count})")
    print("\n  Assertion passed: trajectory count == original count")
    print("=" * 60 + "\n")

    # 计算嵌入
    embeddings = compute_embeddings(model, tokenizer, trajectory_texts, args, device)
    print(f"Embeddings shape: {embeddings.shape}")

    # 再次校验
    assert embeddings.shape[0] == original_count, \
        (f"FATAL: embeddings count ({embeddings.shape[0]}) != "
         f"original count ({original_count})")
    print(f"Final check passed: {embeddings.shape[0]} embeddings == {original_count} samples")

    # 保存
    save_data = {
        "embeddings": embeddings,  # [N, hidden_dim]
        "pooling": args.pooling,
        "model_name": args.model_name_or_path,
        "hidden_dim": embeddings.shape[1],
        "num_samples": embeddings.shape[0],
        # 保存匹配统计信息
        "extraction_stats": {
            "original_count": original_count,
            "matched_count": matched_count,
            "unmatched_count": unmatched_count,
            "unmatched_indices": unmatched_indices,
            "match_rate": matched_count / original_count * 100,
        },
    }
    torch.save(save_data, args.output_path)
    print(f"Saved embeddings to {args.output_path}")


if __name__ == "__main__":
    main()
