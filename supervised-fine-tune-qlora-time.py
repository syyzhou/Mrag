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
from model_time import get_peft_model
import re
import datetime
from train_command_logger import save_training_command

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


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")


@dataclass
class DataArguments:
    dataset: str = field(default=None, metadata={"help": "Path to the training data."})


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
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Whether to remove unused columns from the dataset."},
    )
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
    top_k_routing_strategy: bool = field(default=True, metadata={"help": "是否启用 top-k 路由策略"})
    top_k: int = field(default=2, metadata={"help": "路由时选择的 top-k 值"})
 
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
    use_load_balancing_loss: bool = field(default=True, metadata={"help": "是否使用负载均衡损失"})
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
    use_time_aware_routing: bool = field(default=True, metadata={"help": "是否启用 time"})
    time_embed_dim: int = field(default=64, metadata={"help": "时间编码器维度"})




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

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with time information."""

    def __init__(self, dataset: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(dataset)

        logging.warning("Formatting inputs...")

        if '<question>:' not in list_data_dict[0]["question"]:
            sources = ['<question>:' + example["question"] for example in list_data_dict]
            targets = ['<answer>:' + f"{example['answer']}{tokenizer.eos_token}" for example in list_data_dict]
        else:
            sources = [example["question"] for example in list_data_dict]
            targets = [f"{example['answer']}{tokenizer.eos_token}" for example in list_data_dict]

        # ========== 提取目标时间信息 ==========
        logging.warning("Extracting time information...")
        self.time_info_list = []
        for example in list_data_dict:
            time_info = parse_target_time_from_text(example["question"])
            self.time_info_list.append(time_info)
        
        # 统计时间分布
        hours = [t['hour'] for t in self.time_info_list]
        weekdays = [t['weekday'] for t in self.time_info_list]
        weekends = sum(t['is_weekend'] for t in self.time_info_list)
        logging.warning(f"Time stats - Hour range: {min(hours)}-{max(hours)}, "
                       f"Weekend samples: {weekends}/{len(self.time_info_list)}")
        # ========================================
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        time_info = self.time_info_list[i]
        if time_info is None:
            print(f"Problem at index {i}: {time_info}")
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            hour=self.time_info_list[i]['hour'],
            weekday=self.time_info_list[i]['weekday'],
            is_weekend=self.time_info_list[i]['is_weekend'],
        )

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning with time info."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        print(f"Collator received {len(instances)} instances")
        print(f"First instance type: {type(instances[0])}")
        if hasattr(instances[0], 'keys'):
            print(f"First instance keys: {list(instances[0].keys())}")
            
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        # ========== 新增：收集时间信息 ==========
        hours = torch.tensor([instance['hour'] for instance in instances], dtype=torch.long)
        weekdays = torch.tensor([instance['weekday'] for instance in instances], dtype=torch.long)
        is_weekends = torch.tensor([instance['is_weekend'] for instance in instances], dtype=torch.long)
        # ========================================

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            # ========== 新增：时间字段 ==========
            hour=hours,
            weekday=weekdays,
            is_weekend=is_weekends,
            # ====================================
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, dataset=data_args.dataset)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def parse_target_time_from_text(text: str) -> Dict[str, int]:
    """
    从 question 文本中解析目标时间
    目标时间是最后一个 "At YYYY-MM-DD HH:MM:SS" 格式的时间（紧跟着 "Which POI"）
    
    Args:
        text: question 文本
        
    Returns:
        dict: {'hour': int, 'weekday': int, 'is_weekend': int}
    """
    # 匹配 "At YYYY-MM-DD HH:MM:SS" 格式
    pattern = r'At (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})'
    matches = re.findall(pattern, text)
    
    if matches:
        # 取最后一个时间（即目标时间）
        date_str, time_str = matches[-1]
        target_time_str = f"{date_str} {time_str}"
        dt = datetime.datetime.strptime(target_time_str, '%Y-%m-%d %H:%M:%S')
        
        return {
            'hour': dt.hour,
            'weekday': dt.weekday(),  # 0=Monday, 6=Sunday
            'is_weekend': 1 if dt.weekday() >= 5 else 0
        }
        
class UmraTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 获取标准的模型输出和 loss
        outputs = model(**inputs)
        loss = outputs.loss

        attention_mask = inputs.get("attention_mask", None)

        # 添加自定义的 auxiliary loss
        if hasattr(model, "router_manager") and hasattr(model.router_manager, "get_auxiliary_loss"):
            print(f"Computing auxiliary loss")
            loss = model.router_manager.get_auxiliary_loss(loss, attention_mask)

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        model.train()
        
        # 处理 DataParallel
        actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        
        inputs = self._prepare_inputs(inputs)
        
        # ========== 新增：处理时间信息 ==========
        # 从 inputs 中提取时间字段
        time_fields = ['hour', 'weekday', 'is_weekend']
        time_info = {}
        for field in time_fields:
            if field in inputs:
                time_info[field] = inputs.pop(field)  # 取出并从 inputs 中删除
        
        # 如果存在时间信息且模型有 time_encoder
        # if time_info and hasattr(actual_model, 'time_encoder') and actual_model.time_encoder is not None:
        #     # 编码时间向量
        #     time_vector = actual_model.time_encoder(time_info)
        #     # 设置到所有 adapter 层
        #     actual_model.router_manager.set_time_vector(time_vector)
        # ========================================
        # ✅ 传递原始 time_info（不是 time_vector）
        if time_info and hasattr(actual_model, 'time_encoder') and actual_model.time_encoder is not None:
            # 编码时间向量
            time_vector = actual_model.time_encoder(time_info)
            # 设置到所有 adapter 层
            actual_model.router_manager.set_time_vector(time_vector)
        # ========================================
        
        
        # 计算 loss
        if self.args.fp16 and hasattr(self, 'use_amp') and self.use_amp:
            with self.autocast_smart_context_manager():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        # 反向传播
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        # 清理
        if hasattr(actual_model, "router_manager") and hasattr(actual_model.router_manager, "clear"):
            actual_model.router_manager.clear()

        return loss.detach()


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ConfigArguments))
    model_args, data_args, training_args, config_args = parser.parse_args_into_dataclasses()
    save_training_command(training_args.output_dir, os.path.basename(__file__),
                          model_args=model_args, data_args=data_args,
                          training_args=training_args, config_args=config_args)

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

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    dataset_name = data_args.dataset.split('/')[2]
    poi_num = {'nyc': 4981, 'tky': 7833, 'ca': 9690}[dataset_name]
    peft_config = UmRaConfig(
        target_modules=config_args.target_modules,
        target_modules_lora=config_args.target_modules_lora,
        dropout=config_args.dropout,
        poi_num=poi_num,
        # routing strategy
        top_k_routing_strategy=config_args.top_k_routing_strategy,
        top_k=config_args.top_k,
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
        lora_r=config_args.lora_r,
        lora_alpha=config_args.lora_alpha,
        use_time_aware_routing=config_args.use_time_aware_routing,
        time_embed_dim=config_args.time_embed_dim,
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

    trainer = UmraTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    # 不用额外的loss
    # trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    # trainer.save_state()
    # trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
