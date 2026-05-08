# Written by Peibo Li
# Modified for query embedding extraction

import io
import os
import copy
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from torch.utils.data import DataLoader

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig

from config import TARGET_MODULE_TYPE, UmRaConfig
from model8 import MoraModel, get_peft_model
import re


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
os.environ["WANDB_DISABLED"] = "true"


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen2-1.5B")
    model_type: Optional[str] = field(default="llama")


@dataclass
class DataArguments:
    train_dataset: str = field(default=None, metadata={"help": "Path to the training data."})
    test_dataset: str = field(default=None, metadata={"help": "Path to the test data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output2_dir: str = field(default="./output/umra_model8/")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(default=True)
    use_full_attn: bool = field(default=False)
    low_rank_training: bool = field(default=True)
    trainable_params: str = field(default="embed_tokens,norm")
    batch_size: int = field(default=8, metadata={"help": "Batch size for encoding"})


@dataclass
class ConfigArguments:
    dropout: float = field(default=0.0)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    target_modules: str = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    target_modules_lora: Optional[str] = field(default=None)
    top_k_routing_strategy: bool = field(default=False)
    top_k: int = field(default=2)
    use_task_router: bool = field(default=False)
    task_router_only: bool = field(default=False)
    share_router_for_qkv: bool = field(default=False)
    share_router_for_w_i: bool = field(default=False)
    num_router_mlp_layers: int = field(default=1)
    router_hidden_dim: int = field(default=32)
    epsilon_alpha: float = field(default=2.0)
    alpha_shift: float = field(default=0.0)
    alpha_up_bound: float = field(default=0.8)
    alpha_low_bound: float = field(default=0.2)
    use_load_balancing_loss: bool = field(default=False)
    use_div_loss: bool = field(default=False)
    gamma_div_certain_t: float = field(default=0.5)
    gamma_div_balance_t: float = field(default=0.98)
    gamma_div_certain_s: float = field(default=0.5)
    gamma_div_balance_s: float = field(default=0.98)
    lambda_auxiliary: float = field(default=0.005)
    lambda_lm: float = field(default=1.0)
    eta_b: float = field(default=1.2)
    num_experts: int = field(default=4)
    use_hydra_lora: bool = field(default=True)


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


class QuestionOnlyDataset(Dataset):
    """
    ✅ 只加载 question 用于编码的数据集
    """
    def __init__(self, dataset_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        logging.warning(f"Loading data from {dataset_path}...")
        
        # 支持 json 和 csv 格式
        if dataset_path.endswith('.json'):
            list_data_dict = jload(dataset_path)
        elif dataset_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(dataset_path)
            list_data_dict = df.to_dict('records')
        else:
            # 尝试作为 json 加载
            list_data_dict = jload(dataset_path)
        
        logging.warning(f"Loaded {len(list_data_dict)} samples")
        
        # ✅ 只处理 question 部分
        self.questions = []
        self.input_ids = []
        self.attention_masks = []
        
        for example in tqdm(list_data_dict, desc="Tokenizing questions"):
            question = example["question"]
            
            # 添加前缀（与训练时一致）
            if '<question>:' not in question:
                question = '<question>:' + question
            
            self.questions.append(question)
            
            # Tokenize
            tokenized = tokenizer(
                question,
                return_tensors="pt",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            
            self.input_ids.append(tokenized.input_ids[0])
            self.attention_masks.append(tokenized.attention_mask[0])
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_masks[i]
        }


@dataclass
class DataCollatorForQuestionEncoding:
    """
    ✅ 用于编码 question 的 collator
    """
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        attention_masks = [instance["attention_mask"] for instance in instances]
        
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
        }


def encode_questions(
    model,
    tokenizer,
    dataset_path: str,
    output_path: str,
    device: str,
    batch_size: int = 8,
    dataset_name: str = "train"
):
    """
    ✅ 对问题进行编码并保存
    
    Args:
        model: 模型
        tokenizer: 分词器
        dataset_path: 数据集路径
        output_path: 输出目录
        device: 设备
        batch_size: 批次大小
        dataset_name: 数据集名称（用于文件命名）
    """
    print(f"\n{'='*60}")
    print(f"Encoding {dataset_name} questions...")
    print(f"Input: {dataset_path}")
    print(f"{'='*60}")
    
    # 创建数据集
    dataset = QuestionOnlyDataset(dataset_path, tokenizer)
    data_collator = DataCollatorForQuestionEncoding(tokenizer=tokenizer)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=data_collator,
        num_workers=0
    )
    
    num_samples = len(dataset)
    hidden_dim = model.config.hidden_size
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 输出文件路径
    embedding_file = os.path.join(output_path, f'{dataset_name}_query_embeddings.dat')
    meta_file = os.path.join(output_path, f'{dataset_name}_emb_meta.json')
    
    # 创建 memmap 文件
    fp = np.memmap(embedding_file, dtype='float32', mode='w+', shape=(num_samples, hidden_dim))
    
    print(f"Total samples: {num_samples}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Output file: {embedding_file}")
    
    current_idx = 0
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Encoding {dataset_name}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 前向传播获取隐藏状态
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # 获取最后一层隐藏状态
            last_hidden_state = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
            
            # Mean Pooling（考虑 attention_mask）
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            query_vector = sum_embeddings / sum_mask  # (batch, hidden_dim)
            
            # 写入 memmap
            batch_size_actual = query_vector.size(0)
            fp[current_idx:current_idx + batch_size_actual, :] = query_vector.cpu().numpy()
            current_idx += batch_size_actual
    
    # 刷新到磁盘
    fp.flush()
    del fp
    
    # 保存元信息
    meta_info = {
        "num_samples": num_samples,
        "hidden_dim": hidden_dim,
        "dataset": dataset_name,
        "source_file": dataset_path
    }
    with open(meta_file, 'w') as f:
        json.dump(meta_info, f, indent=2)
    
    print(f"\n✓ Saved {num_samples} embeddings to: {embedding_file}")
    print(f"✓ Saved metadata to: {meta_file}")
    
    return embedding_file, meta_file


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ConfigArguments))
    model_args, data_args, training_args, config_args = parser.parse_args_into_dataclasses()
    
    # ==================== 加载配置 ====================
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir
    )
    
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    
    # ==================== 加载模型 ====================
    print("\n[1/3] Loading base model...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )
    
    # ==================== 加载 tokenizer ====================
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
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
    
    # ==================== 加载训练好的权重 ====================
    print("\n[2/3] Loading trained weights...")
    model = MoraModel.from_pretrained(model, training_args.output2_dir)
    peft_weights_path = os.path.join(training_args.output2_dir, 'adapter_model.safetensors')
    
    if os.path.exists(peft_weights_path):
        peft_weights = torch.load(peft_weights_path)
        model.load_state_dict(peft_weights, strict=False)
        print(f"✓ Loaded weights from: {peft_weights_path}")
    else:
        # 尝试其他格式
        peft_weights_path = os.path.join(training_args.output2_dir, 'adapter_model.bin')
        if os.path.exists(peft_weights_path):
            peft_weights = torch.load(peft_weights_path)
            model.load_state_dict(peft_weights, strict=False)
            print(f"✓ Loaded weights from: {peft_weights_path}")
        else:
            print(f"⚠ Warning: No adapter weights found in {training_args.output2_dir}")
    
    # 移动到设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"✓ Model loaded on {device}")
    
    # ==================== 编码问题 ====================
    print("\n[3/3] Encoding questions...")
    
    # 处理训练集
    if data_args.train_dataset and os.path.exists(data_args.train_dataset):
        encode_questions(
            model=model,
            tokenizer=tokenizer,
            dataset_path=data_args.train_dataset,
            output_path=training_args.output_dir,
            device=device,
            batch_size=training_args.batch_size,
            dataset_name="train"
        )
    else:
        print(f"⚠ Train dataset not found: {data_args.train_dataset}")
    
    # 处理测试集
    if data_args.test_dataset and os.path.exists(data_args.test_dataset):
        encode_questions(
            model=model,
            tokenizer=tokenizer,
            dataset_path=data_args.test_dataset,
            output_path=training_args.output_dir,
            device=device,
            batch_size=training_args.batch_size,
            dataset_name="test"
        )
    else:
        print(f"⚠ Test dataset not found: {data_args.test_dataset}")
    
    print("\n" + "="*60)
    print("✓ All done!")
    print("="*60)


if __name__ == "__main__":
    main()