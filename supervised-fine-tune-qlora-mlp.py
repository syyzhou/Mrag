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
from model8 import MoraModel, get_peft_model
import re
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
    output2_dir: str = field(default="./output/umra_model8/")
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
        default="embed_tokens,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
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
    top_k_routing_strategy: bool = field(default=False, metadata={"help": "是否启用 top-k 路由策略"})
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
    """Dataset for supervised fine-tuning."""

    def __init__(self, dataset: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(dataset)

        logging.warning("Tokenizing inputs... This may take some time...")
        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        for example in list_data_dict:
            text = example["question"]
            encoding = tokenizer(
                text,
                truncation=True,
                padding="longest",
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            )
            self.input_ids.append(encoding["input_ids"][0])
            self.attention_masks.append(encoding["attention_mask"][0])
            answer_str = example["answer"]
            answer_num = answer_str.replace("<answer>: ", "")  # 使用replace去除
            self.labels.append(int(answer_num))
            # self.labels.append(int(example["answer"]))  


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=torch.tensor(self.labels[i], dtype=torch.long), attention_mask=self.attention_masks[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        attention_mask = [instance["attention_mask"] for instance in instances]
        labels = torch.tensor([instance["labels"] for instance in instances], dtype=torch.long)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, dataset=data_args.dataset)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
class UmraTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 获取标准的模型输出和loss
        # outputs = model(**inputs)
        # loss = outputs.loss

        # # 获取attention_mask，如果需要的话
        # attention_mask = inputs.get("attention_mask", None)

        # # 添加自定义的 auxiliary loss
        # if hasattr(model, "router_manager") and hasattr(model.router_manager, "get_auxiliary_loss"):
        #     loss = model.router_manager.get_auxiliary_loss(loss, attention_mask)

        # return (loss, outputs) if return_outputs else loss
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        model.train()
        if isinstance(model, torch.nn.DataParallel):
            print("模型被 DataParallel 包裹了")
            model = model.module  # 解除包裹
        inputs = self._prepare_inputs(inputs)
        
        # if hasattr(model, "task_encoder") and model.task_encoder is not None:
        #     if "input_ids" in inputs:
        #         embedding_fn = getattr(
        #             model.base_model,
        #             TARGET_MODULE_TYPE[model.config.model_type]['embed']
        #         )
        #         hidden_states = embedding_fn(inputs["input_ids"])

        #         task_embed = model.task_encoder(hidden_states, inputs['attention_mask'])
        #         task_embed = task_embed.to(dtype=torch.bfloat16)  
        #         model.router_manager.set_task_weight(task_embed)
        # 梯度缩放（如果使用 fp16）
        if self.args.fp16 and self.use_amp:
            with self.autocast_smart_context_manager():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        # 反向传播
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        if hasattr(model, "router_manager") and hasattr(model.router_manager, "clear"):
            model.router_manager.clear()

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
    [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]


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
    poi_num = {'nyc': 4980, 'tky': 7832, 'ca': 9689}[dataset_name]
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
        lora_alpha=config_args.lora_alpha
    )
    peft_config.torch_dtype = torch.bfloat16
    peft_config.padding_side = tokenizer.padding_side
    
    # model = get_peft_model(model, peft_config)
    
    model = MoraModel.from_pretrained(model, training_args.output2_dir)
    peft_weights = torch.load(training_args.output2_dir + '/' + 'adapter_model.safetensors')
    model.load_state_dict(peft_weights, strict=False)
    
    # 读取配置文件
    with open(training_args.output2_dir+ '/' + 'config.json', 'r') as f:
        config = json.load(f)
  
    class MLPHead(nn.Module):
        def __init__(self, config, input_dim, hidden_dim, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc1 = self.fc1.to(dtype=config['torch_dtype'])  # 将dtype应用到层上
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, num_classes)
            self.fc2 = self.fc2.to(dtype=config['torch_dtype'])  # 将dtype应用到层上
        def forward(self, x):
            x = x.mean(dim=1)  
            return self.fc2(self.relu(self.fc1(x)))
        
    mlp_head = MLPHead(config, config['hidden_size'], config['hidden_size'], poi_num)
    
    if hasattr(model, "lm_head"):
        setattr(model, "lm_head", mlp_head)
        
    for param in model.parameters():
        param.requires_grad = False  
    for param in model.lm_head.parameters():
        param.requires_grad = True  
    # trainer = Trainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     **data_module
    # )
    # class CastOutputToFloat(nn.Sequential):
    #     def forward(self, x):
    #         return super().forward(x).to(torch.float32)

    # model.lm_head = CastOutputToFloat(model.lm_head)

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
