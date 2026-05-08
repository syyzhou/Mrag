import os
import torch
import torch.nn as nn
import transformers
from transformers import Trainer, TrainingArguments as HfTrainingArguments
from dataclasses import dataclass, field
from typing import Optional

from moe_lora import MoELoRAConfig, get_peft_model_moe_lora

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-2-7b-hf")

@dataclass
class MoELoRAArguments:
    lora_r: int = field(default=32, metadata={"help": "Total LoRA rank (must be divisible by num_experts)"})
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    num_experts: int = field(default=4, metadata={"help": "Number of experts"})
    top_k: int = field(default=2, metadata={"help": "Number of activated experts"})
    router_type: str = field(default="sequence", metadata={"help": "token or sequence"})
    load_balance_weight: float = field(default=0.01)

@dataclass  
class TrainingArguments(HfTrainingArguments):
    model_max_length: int = field(default=4096)
    output_dir: str = field(default="./output")


class MoELoRATrainer(Trainer):
    """支持 MoE-LoRA 辅助损失的 Trainer"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # 标准的语言模型损失
        outputs = model(**inputs)
        loss = outputs.loss
        
        # 添加路由负载均衡损失
        if hasattr(model, 'router_manager'):
            aux_loss = model.router_manager.get_auxiliary_loss()
            loss = loss + aux_loss
        
        # 清除 router 缓存
        if hasattr(model, 'router_manager'):
            model.router_manager.clear()
        
        return (loss, outputs) if return_outputs else loss


def train():
    parser = transformers.HfArgumentParser((
        ModelArguments, MoELoRAArguments, TrainingArguments
    ))
    model_args, moe_lora_args, training_args = parser.parse_args_into_dataclasses()
    
    # 创建 MoE-LoRA 配置
    moe_lora_config = MoELoRAConfig(
        lora_r=moe_lora_args.lora_r,
        lora_alpha=moe_lora_args.lora_alpha,
        lora_dropout=moe_lora_args.lora_dropout,
        num_experts=moe_lora_args.num_experts,
        top_k=moe_lora_args.top_k,
        router_type=moe_lora_args.router_type,
        load_balance_weight=moe_lora_args.load_balance_weight,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # 加载模型
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    
    # 应用 MoE-LoRA
    model = get_peft_model_moe_lora(model, moe_lora_config)
    
    # 加载 tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 准备数据集（这里使用你原有的数据加载逻辑）
    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # 使用自定义 Trainer
    trainer = MoELoRATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        # **data_module
    )
    
    trainer.train()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()