# Written by Peibo Li
# Original code based on https://github.com/dvlab-research/LongLoRA?tab=readme-ov-file
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import copy
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
# from llama_attn_replace_sft import replace_llama_attn
# from gptneox_attn_replace import replace_gpt_neox_attn
# from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
from config import TARGET_MODULE_TYPE, UmRaConfig
from model8 import get_peft_model
import re
import seaborn as sns



IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
os.environ["WANDB_DISABLED"]="true"

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


def infer_dataset_name(dataset_path: str, explicit_name: Optional[str] = None) -> str:
    if explicit_name is not None:
        return explicit_name

    path = dataset_path.replace("\\", "/").lower()
    basename = os.path.basename(path)
    for name in ["nyc", "tky", "ca"]:
        if f"/{name}/" in path or path.endswith(f"/{name}") or f"{name}_" in basename:
            return name

    raise ValueError(
        f"Cannot infer dataset_name from path: {dataset_path}. "
        "Please pass --dataset_name nyc|tky|ca"
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")


@dataclass
class DataArguments:
    dataset: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Dataset name: nyc/tky/ca"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default=None,
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )
    remove_unused_columns: bool = field(default=False)
@dataclass
class ConfigArguments:
    dropout: float = field(default=0.0, metadata={"help": "Dropout概率"})
    # LoRA 参数
    lora_r: int = field(default=8, metadata={"help": "LoRA 低秩矩阵的秩"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA 缩放因子"})
    target_modules: str = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
                                      metadata={"help": "LoRA 目标模块"})
    target_modules_lora: Optional[str] = field(default=None, metadata={"help": "LoRA 特定目标模块"})

    # HMORA 路由策略 
    top_k_routing_strategy: bool = field(default=False, metadata={"help": "是否启用 top-k 路由策略"})
    top_k: int = field(default=2, metadata={"help": "路由时选择的 top-k 值"})
    trajectory_top_k_routing_strategy: bool = field(
        default=False, metadata={"help": "是否仅对轨迹路由启用 top-k 稀疏选择"}
    )
    trajectory_top_k: int = field(
        default=2, metadata={"help": "轨迹路由选择的 top-k 值"}
    )
 
    # HMORA 路由共享相关
    use_task_router: bool = field(default=False, metadata={"help": "是否使用任务路由器"})
    task_router_only: bool = field(default=False, metadata={"help": "是否仅使用任务路由器"})
    share_router_for_qkv: bool = field(default=False, metadata={"help": "是否共享 QKV 路由器"})
    share_router_for_w_i: bool = field(default=False, metadata={"help": "是否共享 W_i 路由器"})

    # HMORA 路由配置
    num_router_mlp_layers: int = field(default=1, metadata={"help": "路由器 MLP 层数"})
    router_hidden_dim: int = field(default=32, metadata={"help": "路由器隐藏层维度"})
    epsilon_alpha: float = field(default=2.0, metadata={"help": "epsilon alpha 超参数"})
    alpha_shift: float = field(default=0.0, metadata={"help": "alpha 偏移"})
    alpha_up_bound: float = field(default=0.8, metadata={"help": "alpha 上限"})
    alpha_low_bound: float = field(default=0.2, metadata={"help": "alpha 下限"})

    # HMORA 损失项
    use_load_balancing_loss: bool = field(default=False, metadata={"help": "是否使用负载均衡损失"})
    use_div_loss: bool = field(default=False, metadata={"help": "是否使用多样性损失"})
    gamma_div_certain_t: float = field(default=0.5, metadata={"help": "γ 多样性确定性（任务）"})
    gamma_div_balance_t: float = field(default=0.98, metadata={"help": "γ 多样性平衡性（任务）"})
    gamma_div_certain_s: float = field(default=0.5, metadata={"help": "γ 多样性确定性（样本）"})
    gamma_div_balance_s: float = field(default=0.98, metadata={"help": "γ 多样性平衡性（样本）"})
    lambda_auxiliary: float = field(default=0.005, metadata={"help": "辅助损失权重"})
    lambda_lm: float = field(default=1.0, metadata={"help": "语言建模损失权重"})

    # HMORA Experts
    eta_b: float = field(default=1.2, metadata={"help": "专家路由冗余率"})
    num_experts: int = field(default=4, metadata={"help": "专家数量"})
    use_hydra_lora: bool = field(default=True, metadata={"help": "是否启用 HydraLoRA"})
    router1_use_shared_expert: bool = field(default=False, metadata={"help": "是否为router1启用共享专家"})
    router1_shared_expert_weight: float = field(default=1.0, metadata={"help": "router1共享专家权重"})
    
    # 轨迹编码路由
    use_trajectory_routing: bool = field(
        default=False, metadata={"help": "是否使用预编码轨迹向量辅助路由"}
    )
    trajectory_embedding_path: Optional[str] = field(
        default=None, metadata={"help": "预编码轨迹向量的路径 (.pt 或 .npy)"}
    )
    trajectory_fusion_mode: str = field(
        default="gate", metadata={"help": "轨迹向量融合方式: gate/add/concat_proj/cross_attention"}
    )
    share_traj_projector: bool = field(
        default=True, metadata={"help": "是否所有层共享轨迹投影器"}
    )
    





def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


# class SupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(self, dataset: str, tokenizer: transformers.PreTrainedTokenizer):
#         super(SupervisedDataset, self).__init__()
#         logging.warning("Loading data...")
#         list_data_dict = jload(dataset)

#         logging.warning("Formatting inputs...")

#         if '<question>:' not in list_data_dict[0]["question"]:
#             sources = ['<question>:' + example["question"] for example in list_data_dict]
#             targets = ['<answer>:' + f"{example['answer']}{tokenizer.eos_token}" for example in list_data_dict]
#         else:
#             sources = [example["question"] for example in list_data_dict]
#             targets = [f"{example['answer']}{tokenizer.eos_token}" for example in list_data_dict]
#         logging.warning("Tokenizing inputs... This may take some time...")
#         data_dict = preprocess(sources, targets, tokenizer)

#         self.input_ids = data_dict["input_ids"]
#         self.labels = data_dict["labels"]

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         return dict(input_ids=self.input_ids[i], labels=self.labels[i])

# 在 sft.py 中修改 SupervisedDataset 和 DataCollator

class SupervisedDatasetWithEmbeddings(Dataset):
    """带有预编码轨迹向量的监督微调数据集"""

    def __init__(self, dataset: str, tokenizer: transformers.PreTrainedTokenizer,
                 trajectory_embedding_path: Optional[str] = None):
        super(SupervisedDatasetWithEmbeddings, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(dataset)

        logging.warning("Formatting inputs...")
        if '<question>:' not in list_data_dict[0]["question"]:
            sources = ['<question>:' + example["question"] for example in list_data_dict]
            targets = ['<answer>:' + f"{example['answer']}{tokenizer.eos_token}" for example in list_data_dict]
        else:
            sources = [example["question"] for example in list_data_dict]
            targets = [f"{example['answer']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        # 加载预编码的轨迹向量
        self.trajectory_embeddings = None
        if trajectory_embedding_path is not None and os.path.exists(trajectory_embedding_path):
            logging.warning(f"Loading trajectory embeddings from {trajectory_embedding_path}...")
            if trajectory_embedding_path.endswith(".pt"):
                embed_data = torch.load(trajectory_embedding_path, map_location="cpu")
                self.trajectory_embeddings = embed_data["embeddings"]  # [N, hidden_dim]
            elif trajectory_embedding_path.endswith(".npy"):
                self.trajectory_embeddings = torch.from_numpy(
                    np.load(trajectory_embedding_path)
                )
            
            assert len(self.trajectory_embeddings) == len(self.input_ids), \
                f"Embedding数量({len(self.trajectory_embeddings)})与样本数量({len(self.input_ids)})不匹配"
            logging.warning(f"Loaded {len(self.trajectory_embeddings)} trajectory embeddings, "
                          f"dim={self.trajectory_embeddings.shape[1]}")
        else:
            logging.warning("No trajectory embeddings loaded, router will use token hidden states only.")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        result = dict(input_ids=self.input_ids[i], labels=self.labels[i])
        if self.trajectory_embeddings is not None:
            result["trajectory_embedding"] = self.trajectory_embeddings[i]
        return result


@dataclass
class DataCollatorWithEmbeddings(object):
    """支持轨迹向量的DataCollator"""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        result = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # 如果有轨迹向量，也打包进去
        if "trajectory_embedding" in instances[0]:
            trajectory_embeddings = torch.stack(
                [inst["trajectory_embedding"] for inst in instances], dim=0
            )
            result["trajectory_embedding"] = trajectory_embeddings

        return result


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    trajectory_embedding_path: Optional[str] = None,
) -> Dict:
    train_dataset = SupervisedDatasetWithEmbeddings(
        tokenizer=tokenizer,
        dataset=data_args.dataset,
        trajectory_embedding_path=trajectory_embedding_path,
    )
    data_collator = DataCollatorWithEmbeddings(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

# @dataclass
# class DataCollatorForSupervisedDataset(object):
#     """Collate examples for supervised fine-tuning."""

#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
#         return dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )


# def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
#     """Make dataset and collator for supervised fine-tuning."""
#     train_dataset = SupervisedDataset(tokenizer=tokenizer, dataset=data_args.dataset)
#     data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
#     return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

# class UmraTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         # 获取标准的模型输出和loss
#         outputs = model(**inputs)
#         loss = outputs.loss

#         # 获取attention_mask，如果需要的话
#         attention_mask = inputs.get("attention_mask", None)

#         # 添加自定义的 auxiliary loss
#         if hasattr(model, "router_manager") and hasattr(model.router_manager, "get_auxiliary_loss"):
#             loss = model.router_manager.get_auxiliary_loss(loss, attention_mask)

#         return (loss, outputs) if return_outputs else loss

#     def training_step(self, model, inputs):
#         model.train()
#         if isinstance(model, torch.nn.DataParallel):
#             print("模型被 DataParallel 包裹了")
#             model = model.module  # 解除包裹
#         inputs = self._prepare_inputs(inputs)
        
#         # if hasattr(model, "task_encoder") and model.task_encoder is not None:
#         #     if "input_ids" in inputs:
#         #         embedding_fn = getattr(
#         #             model.base_model,
#         #             TARGET_MODULE_TYPE[model.config.model_type]['embed']
#         #         )
#         #         hidden_states = embedding_fn(inputs["input_ids"])

#         #         task_embed = model.task_encoder(hidden_states, inputs['attention_mask'])
#         #         task_embed = task_embed.to(dtype=torch.bfloat16)  
#         #         model.router_manager.set_task_weight(task_embed)
#         # 梯度缩放（如果使用 fp16）
#         if self.args.fp16 and self.use_amp:
#             with self.autocast_smart_context_manager():
#                 loss = self.compute_loss(model, inputs)
#         else:
#             loss = self.compute_loss(model, inputs)

#         # 反向传播
#         if self.args.gradient_accumulation_steps > 1:
#             loss = loss / self.args.gradient_accumulation_steps

#         self.accelerator.backward(loss)

#         if hasattr(model, "router_manager") and hasattr(model.router_manager, "clear"):
#             model.router_manager.clear()

#         return loss.detach()
    
# sft.py 中修改 UmraTrainer

# class UmraTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         # 从inputs中取出轨迹向量（如果有的话）
#         trajectory_embedding = inputs.pop("trajectory_embedding", None)
        
#         # 在forward之前，将轨迹向量分发到所有router
#         if trajectory_embedding is not None and hasattr(model, "router_manager"):
#             trajectory_embedding = trajectory_embedding.to(
#                 dtype=model.dtype if hasattr(model, 'dtype') else torch.bfloat16,
#                 device=next(model.parameters()).device
#             )
#             for router in model.router_manager.token_routers:
#                 if hasattr(router, 'set_trajectory_embedding'):
#                     router.set_trajectory_embedding(trajectory_embedding)
        
#         outputs = model(**inputs)
#         loss = outputs.loss

#         attention_mask = inputs.get("attention_mask", None)
#         if hasattr(model, "router_manager") and hasattr(model.router_manager, "get_auxiliary_loss"):
#             loss = model.router_manager.get_auxiliary_loss(loss, attention_mask)

#         return (loss, outputs) if return_outputs else loss
    
# class UmraTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         # 从inputs中取出轨迹向量
#         trajectory_embedding = inputs.pop("trajectory_embedding", None)
        
#         # 分发轨迹向量到 use_trajectory=True 的router（即router2）
#         if trajectory_embedding is not None and hasattr(model, "router_manager"):
#             trajectory_embedding = trajectory_embedding.to(
#                 dtype=torch.bfloat16,
#                 device=next(model.parameters()).device
#             )
#             for router in model.router_manager.token_routers:
#                 if getattr(router, 'use_trajectory', False):
#                     router.set_trajectory_embedding(trajectory_embedding)
        
#         outputs = model(**inputs)
#         loss = outputs.loss

#         attention_mask = inputs.get("attention_mask", None)
#         if hasattr(model, "router_manager") and hasattr(model.router_manager, "get_auxiliary_loss"):
#             loss = model.router_manager.get_auxiliary_loss(loss, attention_mask)

#         # 清理router缓存
#         if hasattr(model, "router_manager") and hasattr(model.router_manager, "clear"):
#             model.router_manager.clear()

#         return (loss, outputs) if return_outputs else loss

class UmraTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        trajectory_embedding = inputs.pop("trajectory_embedding", None)

        if trajectory_embedding is not None:
            trajectory_embedding = trajectory_embedding.to(
                dtype=torch.bfloat16,
                device=next(model.parameters()).device
            )
            seen = set()
            if hasattr(model, "router_manager"):
                for router in model.router_manager.token_routers:
                    if getattr(router, "use_trajectory", False) and router.traj_projector is not None:
                        pid = id(router.traj_projector)
                        if pid not in seen:
                            router.traj_projector.set_trajectory_embedding(trajectory_embedding)
                            seen.add(pid)

            # ── 只设置到 traj_projector 上，所有router2共享 ──────
            # 梯度检查点重新forward时，这里的值仍然存在
            # if hasattr(model, 'traj_projector') and model.traj_projector is not None:
            #     model.traj_projector.set_trajectory_embedding(trajectory_embedding)

        outputs = model(**inputs)
        loss = outputs.loss

        attention_mask = inputs.get("attention_mask", None)
        if hasattr(model, "router_manager") and hasattr(model.router_manager, "get_auxiliary_loss"):
            loss = model.router_manager.get_auxiliary_loss(loss, attention_mask)

        # clear 只清 routing_weight，不清 traj_embedding
        if hasattr(model, "router_manager"):
            model.router_manager.clear()
        # # traj_embedding 在 compute_loss 结束后清除
        # if hasattr(model, 'traj_projector') and model.traj_projector is not None:
        #     model.traj_projector.clear()

        return (loss, outputs) if return_outputs else loss
    
    # 不再重写 training_step，让 Trainer 自己处理

    # def training_step(self, model, inputs):
    #     model.train()
    #     if isinstance(model, torch.nn.DataParallel):
    #         model = model.module
    #     inputs = self._prepare_inputs(inputs)

    #     if self.args.fp16 and self.use_amp:
    #         with self.autocast_smart_context_manager():
    #             loss = self.compute_loss(model, inputs)
    #     else:
    #         loss = self.compute_loss(model, inputs)

    #     if self.args.gradient_accumulation_steps > 1:
    #         loss = loss / self.args.gradient_accumulation_steps

    #     self.accelerator.backward(loss)

    #     if hasattr(model, "router_manager") and hasattr(model.router_manager, "clear"):
    #         model.router_manager.clear()

    #     return loss.detach()

def visualize_all_router_stats(model, save_path='router_analysis'):
    """
    训练结束后生成完整的可视化报告
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # 收集所有统计数据
    token_router_stats = []
    gate_router_stats = []
    
    for router in model.router_manager.token_routers:
        token_router_stats.append(router.get_stats())
    
    for router in model.router_manager.routers:
        gate_router_stats.append(router.get_stats())
    
    # ========== 1. 专家使用频次热力图 ==========
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 提取数据
    layers = sorted(list(set([s['layer_id'] for s in token_router_stats])))
    num_layers = len(layers)
    
    freq_g1 = np.zeros((num_layers, 4))
    freq_g2 = np.zeros((num_layers, 4))
    
    for stat in token_router_stats:
        layer_idx = layers.index(stat['layer_id'])
        if '_G1' in stat['tag']:
            freq_g1[layer_idx] = stat['freq_ratio']
        elif '_G2' in stat['tag']:
            freq_g2[layer_idx] = stat['freq_ratio']
    
    # 绘制G1热力图
    sns.heatmap(freq_g1, annot=True, fmt='.2f', cmap='YlOrRd', 
                xticklabels=[f'E{i}' for i in range(4)],
                yticklabels=[f'L{i}' for i in layers],
                vmin=0, vmax=0.5, ax=axes[0], cbar_kws={'label': 'Selection Frequency'})
    axes[0].set_title('Group 1 - Expert Selection Frequency', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Expert ID')
    axes[0].set_ylabel('Layer ID')
    
    # 绘制G2热力图
    sns.heatmap(freq_g2, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=[f'E{i}' for i in range(4)],
                yticklabels=[f'L{i}' for i in layers],
                vmin=0, vmax=0.5, ax=axes[1], cbar_kws={'label': 'Selection Frequency'})
    axes[1].set_title('Group 2 - Expert Selection Frequency', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Expert ID')
    axes[1].set_ylabel('Layer ID')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/expert_selection_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}/expert_selection_heatmap.png")
    plt.close()
    
    # ========== 2. 专家平均权重热力图 ==========
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    weight_g1 = np.zeros((num_layers, 4))
    weight_g2 = np.zeros((num_layers, 4))
    
    for stat in token_router_stats:
        layer_idx = layers.index(stat['layer_id'])
        if '_G1' in stat['tag']:
            weight_g1[layer_idx] = stat['avg_weight']
        elif '_G2' in stat['tag']:
            weight_g2[layer_idx] = stat['avg_weight']
    
    sns.heatmap(weight_g1, annot=True, fmt='.3f', cmap='Reds',
                xticklabels=[f'E{i}' for i in range(4)],
                yticklabels=[f'L{i}' for i in layers],
                ax=axes[0], cbar_kws={'label': 'Avg Weight'})
    axes[0].set_title('Group 1 - Expert Average Weight', fontsize=14, fontweight='bold')
    
    sns.heatmap(weight_g2, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=[f'E{i}' for i in range(4)],
                yticklabels=[f'L{i}' for i in layers],
                ax=axes[1], cbar_kws={'label': 'Avg Weight'})
    axes[1].set_title('Group 2 - Expert Average Weight', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/expert_weight_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}/expert_weight_heatmap.png")
    plt.close()
    
    # ========== 3. 组间Gate使用对比 ==========
    gate_weights = np.zeros((num_layers, 2))
    for stat in gate_router_stats:
        layer_idx = layers.index(stat['layer_id'])
        gate_weights[layer_idx] = stat['group_weights']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(num_layers)
    width = 0.35
    
    ax.bar(x - width/2, gate_weights[:, 0], width, label='Group 1', color='coral')
    ax.bar(x + width/2, gate_weights[:, 1], width, label='Group 2', color='skyblue')
    
    ax.set_xlabel('Layer ID', fontsize=12)
    ax.set_ylabel('Average Gate Weight', fontsize=12)
    ax.set_title('Group Gate Usage Across Layers', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in layers])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/group_gate_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}/group_gate_comparison.png")
    plt.close()
    
    # ========== 4. 异常检测报告 ==========
    with open(f'{save_path}/analysis_report.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ROUTER USAGE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # 检测专家坍缩
        f.write("🔍 Expert Collapse Detection:\n")
        f.write("-" * 80 + "\n")
        for stat in token_router_stats:
            min_freq = np.min(stat['freq_ratio'])
            max_freq = np.max(stat['freq_ratio'])
            if max_freq > 0.7:
                f.write(f"⚠️  {stat['tag']}: Expert {np.argmax(stat['freq_ratio'])} dominates ({max_freq:.1%})\n")
            if min_freq < 0.05:
                f.write(f"⚠️  {stat['tag']}: Expert {np.argmin(stat['freq_ratio'])} underused ({min_freq:.1%})\n")
        f.write("\n")
        
        # 检测组间不平衡
        f.write("🔍 Group Imbalance Detection:\n")
        f.write("-" * 80 + "\n")
        for stat in gate_router_stats:
            g1, g2 = stat['group_weights']
            if abs(g1 - g2) > 0.3:
                f.write(f"⚠️  {stat['tag']}: Imbalanced [G1:{g1:.2f}, G2:{g2:.2f}]\n")
        f.write("\n")
        
        # 详细统计
        f.write("📊 Detailed Statistics:\n")
        f.write("-" * 80 + "\n")
        for stat in token_router_stats:
            f.write(f"\n{stat['tag']}:\n")
            f.write(f"  Selection Freq: {[f'{x:.3f}' for x in stat['freq_ratio']]}\n")
            f.write(f"  Avg Weight:     {[f'{x:.3f}' for x in stat['avg_weight']]}\n")
            f.write(f"  Total Selects:  {stat['select_count'].tolist()}\n")
    
    print(f"✅ Saved: {save_path}/analysis_report.txt")
    print("\n" + "=" * 80)
    print("📊 Visualization Complete! Check the following files:")
    print(f"  - {save_path}/expert_selection_heatmap.png")
    print(f"  - {save_path}/expert_weight_heatmap.png")
    print(f"  - {save_path}/group_gate_comparison.png")
    print(f"  - {save_path}/analysis_report.txt")
    print("=" * 80 + "\n")
    
from transformers import TrainerCallback
import os
import torch
import json

class LoRACheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        model = kwargs["model"]
        optimizer = kwargs["optimizer"]

        save_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(save_dir, exist_ok=True)

        # 1️⃣ 只保存可训练参数（LoRA）
        trainable_params = {
            k: v.detach().cpu()
            for k, v in model.named_parameters()
            if v.requires_grad
        }

        torch.save(
            {
                "model": trainable_params,
                "optimizer": optimizer.state_dict(),
                "step": state.global_step,
                "peft_config": model.peft_config.export(),
            },
            os.path.join(save_dir, "adapter_ckpt.pt"),
        )

        return control

    
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ConfigArguments))
    model_args, data_args, training_args, config_args = parser.parse_args_into_dataclasses()

    # NOTE: May expand supported model types in the future
    # if model_args.model_type == "gpt-neox":
    #     replace_gpt_neox_attn(training_args.use_flash_attn, training_args.use_full_attn)
    # else:
    #     replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     llm_int8_threshold=6.0,
        #     llm_int8_has_fp16_weight=False,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # ),
    )

    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)
    # [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]


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

    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # ===== 使用带轨迹向量的数据集 =====
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        trajectory_embedding_path=config_args.trajectory_embedding_path,
    )
    dataset_name = infer_dataset_name(data_args.dataset, data_args.dataset_name)
    poi_num = {'nyc': 4981, 'tky': 7833, 'ca': 9690}[dataset_name]
    print(f"Using dataset_name={dataset_name}, poi_num={poi_num}")
    peft_config = UmRaConfig(
        target_modules=config_args.target_modules,
        target_modules_lora=config_args.target_modules_lora,
        dropout=config_args.dropout,
        # poi_num=poi_num,
        # routing strategy
        top_k_routing_strategy=config_args.top_k_routing_strategy,
        top_k=config_args.top_k,
        trajectory_top_k_routing_strategy=config_args.trajectory_top_k_routing_strategy,
        trajectory_top_k=config_args.trajectory_top_k,
        # router sharing
        use_task_router=config_args.use_task_router,
        task_router_only=config_args.task_router_only,
        share_router_for_qkv=config_args.share_router_for_qkv,
        share_router_for_w_i=config_args.share_router_for_w_i,
        # router
        num_router_mlp_layers=config_args.num_router_mlp_layers,
        router_hidden_dim=config_args.router_hidden_dim,
        epsilon_alpha=config_args.epsilon_alpha,
        alpha_shift=config_args.alpha_shift,
        alpha_up_bound=config_args.alpha_up_bound,
        alpha_low_bound=config_args.alpha_low_bound,
        # loss
        use_load_balancing_loss=config_args.use_load_balancing_loss,
        use_div_loss=config_args.use_div_loss,
        gamma_div_certain_t=config_args.gamma_div_certain_t,
        gamma_div_balance_t=config_args.gamma_div_balance_t,
        gamma_div_certain_s=config_args.gamma_div_certain_s,
        gamma_div_balance_s=config_args.gamma_div_balance_s,
        lambda_lm=config_args.lambda_lm,
        lambda_auxiliary=config_args.lambda_auxiliary,
        # experts
        num_experts=config_args.num_experts,
        use_hydra_lora=config_args.use_hydra_lora,
        router1_use_shared_expert=config_args.router1_use_shared_expert,
        router1_shared_expert_weight=config_args.router1_shared_expert_weight,
        lora_r=config_args.lora_r,
        lora_alpha=config_args.lora_alpha,
        use_trajectory_routing=config_args.use_trajectory_routing,
        trajectory_dim=model.config.hidden_size,  # 用同一个LLM编码，维度一致
        trajectory_fusion_mode=config_args.trajectory_fusion_mode,
        share_traj_projector=config_args.share_traj_projector,
        )
    peft_config.torch_dtype = torch.bfloat16
    peft_config.padding_side = tokenizer.padding_side
    
    model = get_peft_model(model, peft_config)
    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing

    # trainer = UmraTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
     # ===== 使用UmraTrainer =====
    trainer = UmraTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        **data_module,
        callbacks=[LoRACheckpointCallback()]
    )
    # 不用额外的loss
    
    # trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module, callbacks=[LoRACheckpointCallback()])
    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    # trainer.save_state()
    # trainer.save_model(output_dir=training_args.output_dir)
    # 训练完成后
    # visualize_all_router_stats(model, save_path='./router_analysis')


if __name__ == "__main__":
    train()
