from transformers import BertTokenizer
import transformers
import io
import os
import copy
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
# from llama_attn_replace_sft import replace_llama_attn
# from gptneox_attn_replace import replace_gpt_neox_attn
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
from RAG.POIIndex import POIIndexer
from eval_next_poi_loss_desc import find_poi_id_from_gt
from config import TARGET_MODULE_TYPE, UmRaConfig
from model_lora import MoraModel
import re
import torch.nn.functional as F

model_name_or_path='./Qwen2-1.5B'
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
os.environ["WANDB_DISABLED"]="true"

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

config = transformers.AutoConfig.from_pretrained(
    model_name_or_path,
)
model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
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
            
tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=4096,
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

# 原始文本
sentence =  "Located on 3rd Avenue in Manhattan, New York City, this department store serves as a central shopping hub in a bustling commercial district. It caters primarily to office workers and local residents who need comprehensive clothing and household items. The area's high foot traffic ensures frequent visits, making it a go-to spot for daily shopping needs."

# 使用分词器编码句子，得到 token_ids
token_ids = tokenizer.encode(sentence, add_special_tokens=True)  # 添加特殊token，如 [CLS], [SEP]

# 输出 token_ids 和它们的对应文本
print(f"Token IDs: {token_ids}")

# 使用 decode 将 token_ids 转换回文本
decoded_text = tokenizer.decode(token_ids[-50:], skip_special_tokens=True)  # skip_special_tokens=True 去除特殊token
print(f"Decoded Text: {decoded_text}")
