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
from typing import Dict, List, Optional, Sequence

import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from tqdm import tqdm
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
from transformers import Qwen2ForCausalLM, Qwen2Config
from train_command_logger import save_training_command
from typing import Optional, Dict
from transformers import AdamW

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
    hidden_size: int = field(default=1536, metadata={"help": "Transformer 模型的隐藏层大小"})
    num_labels: int = field(default=4980, metadata={"help": "poi 数量"})


@dataclass
class DataArguments:
    dataset: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_dataset: str = field(default=None, metadata={"help": "Path to the test data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    output2_dir: str = field(default="./output/umra_traj/")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    seq_len: int = field(
        default=4096
    )
    batch_size: int = field(default=4, metadata={"help": "Batch size for training."})
    learning_rate: float = field(default=1e-5, metadata={"help": "Learning rate for the optimizer."})
    num_train_epochs: int = field(default=50, metadata={"help": "Number of epochs to train for."})
    warmup_steps: int = field(default=10, metadata={"help": "Number of steps for the warmup phase."})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": "Number of gradient accumulation steps."})
    save_steps: int = field(default=500, metadata={"help": "How often to save checkpoints."})
    early_stopping_patience: int = field(default=3, metadata={"help": "Number of epochs with no improvement after which training will be stopped."})
    lr_scheduler_type: str = field(default="linear", metadata={"help": "Type of scheduler to use. Options are 'linear', 'cosine', etc."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon parameter for Adam optimizer."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay to apply (if any)."})


class ClassificationHead(nn.Module):
    def __init__(self, config, model, sentence_bert_dim=768):
        super().__init__()
        self.qwen2_model = model
        for param in self.qwen2_model.parameters():
            param.requires_grad = False  # 冻结大模型的所有参数
        self.classifier = nn.ModuleDict({
            # "cross_attention": nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=8, batch_first=True, dropout=0.1),
            "cross_attention": nn.MultiheadAttention(embed_dim=sentence_bert_dim, num_heads=1, batch_first=True),
            "linear_bert": nn.Linear(config.hidden_size, sentence_bert_dim),
            "layer_norm_1": nn.LayerNorm(sentence_bert_dim),
            # "layer_norm_2": nn.LayerNorm(sentence_bert_dim),
            # "feed_forward": nn.Sequential(
            #     nn.Linear(config.hidden_size, config.hidden_size * 4),
            #     nn.GELU(),
            #     nn.Dropout(0.1),
            #     nn.Linear(config.hidden_size * 4, config.hidden_size),
            #     nn.Dropout(0.1)
            # ),
            "final_fc": nn.Sequential(
                nn.Linear(sentence_bert_dim, sentence_bert_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(sentence_bert_dim // 2, config.num_labels if hasattr(config, 'num_labels') else 4980)
            ),
        })
        
        # 权重初始化 (Weight Initialization)
        self._init_weights()
    
    def _init_weights(self):
        """初始化新增层的权重"""
        for module in [self.classifier["cross_attention"], self.classifier["final_fc"], self.classifier["linear_bert"]]:
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # labels: Optional[torch.Tensor] = None,
        answers: Optional[torch.Tensor] = None,
        retrieval: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            last_poi_ids: 最后一个POI ID列表 [batch_size]
            retrieval_vectors: 棪索向量 [batch_size, num_retrievals, hidden_size]
        """
        
        self.qwen2_model.eval()  # 切换到评估模式
        # 1. 原始大模型前向传播
        outputs = self.qwen2_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # 2. 获取最后一个token的隐藏状态
        # hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        # last_token_hidden = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        last_token_hidden = last_hidden_state[:, -1, :].to(torch.float32)   # [batch_size, hidden_size]
        
        last_token_hidden = self.classifier["linear_bert"](last_token_hidden)  # [batch_size, 768] -> [batch_size, model_output_dim]
        # 3. 准备跨注意力的query, key, value
        query = last_token_hidden.unsqueeze(1)  #
        
        if retrieval is not None:
            # 如果有检索向量，使用检索向量作为key和value
            key = value = retrieval.unsqueeze(1)   # [batch_size, num_retrievals, hidden_size]
        
        # 4. 应用跨注意力机制
        attn_output, attn_weights = self.classifier["cross_attention"](
            query=query,
            key=key,
            value=value,
            need_weights=True
        )
        
        # 5. 残差连接和层归一化
        attn_output = self.classifier["layer_norm_1"](query + attn_output)
        
        # 6. 前馈网络
        # ff_output = self.classifier['feed_forward'](attn_output)
        # ff_output = self.classifier['layer_norm_2'](attn_output + ff_output)
        
        # 7. 分类头
        classification_logits = self.classifier['final_fc'](attn_output.squeeze(1))  # [batch_size, num_classes]
        
        result = {}
        result = {'logits': classification_logits}  # 返回 logits
        if answers is not None:
            # 计算分类任务的损失 (如果存在标签)
            classification_loss = nn.functional.cross_entropy(classification_logits, answers)
            result['loss'] = classification_loss
        
        return result



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
    input_ids = sources_tokenized["input_ids"]
    # labels = copy.deepcopy(input_ids)
    # for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
    #     label[:source_len] = IGNORE_INDEX
    # return dict(input_ids=input_ids, labels=labels)
    return dict(input_ids=input_ids)

def extract_poi_info_from(sources: List[str]) -> List[Dict]:
    """
    批量从所有source文本中提取POI相关信息
    输入: sources列表, 每个元素是一个样本的文本
    输出: poi_info列表, 每个元素是一个样本的POI信息字典
    """
    poi_info_list = []
    
    for i, source in enumerate(tqdm(sources, desc="提取POI信息")):
        try:
            # 提取用户ID
            user_match = re.search(r'user\s+(\d+)', source)
            user_id = user_match.group(1) if user_match else None
            
            # 提取最后一次时间
            last_time_match = re.search(r'At\s+([^,]+),\s+Which\s+POI', source)
            last_time = last_time_match.group(1).strip() if last_time_match else None
            
            # 提取POI IDs
            poi_pattern = r'visited POI id (\d+)'
            poi_matches = re.findall(poi_pattern, source)
            last_poi_id = poi_matches[-1] if poi_matches else None
            
             # 提取POI类别名称和类别ID（假设格式类似于 "which is a <PoiCategoryName> with Category id <PoiCategoryId>"）
            category_name_match = re.findall(r'which\s+is\s+a\s+([\w\s]+)\s+with', source)
            if category_name_match:
                poi_category_name = category_name_match[-1].strip()
            else:
                poi_category_name = None
                
            poi_info = {
                'user': user_id,
                'last_time': last_time,
                'last_poi_id': last_poi_id,
                'poi_category_name': poi_category_name, 
                'source_index': i  # 可选：记录原始索引
            }
            
            poi_info_list.append(poi_info)
            
        except Exception as e:
            print(f"Error extracting POI info from source {i}: {e}")
            # 添加一个空的POI信息
            poi_info_list.append({
                'user': None,
                'last_time': None, 
                'last_poi_id': None,
                'source_index': i
            })
    
    return poi_info_list
    
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, dataset: str, tokenizer: transformers.PreTrainedTokenizer, train: bool):
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
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        poi_info_list = extract_poi_info_from(sources)
        answer_poi_ids = [re.search(r'POI id\s*(\d+)', target).group(1) if re.search(r'POI id\s*(\d+)', target) else None for target in targets]
        
        self.input_ids = data_dict["input_ids"]
        # self.labels = data_dict["labels"]
        self.poi_info = poi_info_list
        self.answer_ids = [
            torch.tensor([int(poi_id)], dtype=torch.long) if poi_id is not None else torch.tensor([-1], dtype=torch.long)
            for poi_id in answer_poi_ids
        ]
        invalid_values = [tensor.item() for tensor in self.answer_ids if not (0 <= tensor.item() <= 4979)]
        assert len(invalid_values) == 0, f"Found {len(invalid_values)} invalid values: {invalid_values}"

        logging.warning("Loading auxiliary files...")
        self.user_transition_df = pd.read_csv("./datasets/nyc/preprocessed/user_poi_transition_counts_top_5.csv")
        self.all_transition_df = pd.read_csv("./datasets/nyc/preprocessed/all_time_period_poi_transition_counts_top_10.csv")
        self.poi_nearest_df = pd.read_csv("./datasets/nyc/preprocessed/poi_nearest_10.csv")
        self.poi_transition_df = pd.read_csv("./datasets/nyc/preprocessed/poi_transition_top_10.csv")
        if train:
            self.time_period_df = pd.read_csv("./datasets/nyc/preprocessed/train_sample_with_time_periods.csv")
        else:self.time_period_df = pd.read_csv("./datasets/nyc/preprocessed/test_sample_with_time_periods.csv")
        
        # 初始化 Sentence-BERT 模型
        logging.warning("Loading Sentence-BERT encoder...")
        self.sentence_encoder =  SentenceTransformer('/data/ChenWei/zhousiyu/LLM4POI/sentence-transformers/all-mpnet-base-v2')
    
        # 批量处理：一次性准备所有样本的检索信息
        self.retrieved_vecs = []
        retrieved_texts = []
        for i in range(len(self.input_ids)):
            user_id = int(self.poi_info[i]['user'])
            last_poi_id = int(self.poi_info[i]['last_poi_id'])
            last_time = self.poi_info[i]['last_time'].strip('"\'')
            poi_category_name = self.poi_info[i]['poi_category_name']
            
            # ========== 0. 获取全局的类别转移情况 ==========
            self.time_period_df['UTCTimeOffset_str'] = self.time_period_df['UTCTimeOffset'].astype(str)
            time_period_record = self.time_period_df[self.time_period_df['UTCTimeOffset_str'] == last_time]
            time_period = time_period_record['time_period'].values[0] if not time_period_record.empty else None
            transition_data = self.all_transition_df[
                (self.all_transition_df['previous_PoiCategoryName'] == poi_category_name) &
                (self.all_transition_df['time_period'] == time_period)
            ]
            all_trans_text = ', '.join([
                f"{row['previous_PoiCategoryName']} → {row['PoiCategoryName']} ({row['transition_count']})"
                for _, row in transition_data.iterrows()
            ]) or "No transition data for this time period"

            # ========== 1. 获取该用户的类别转移情况 ==========
            user_trans = self.user_transition_df[self.user_transition_df['UserId'] == user_id]
            user_trans_text = ', '.join([
                f"{row['previous_PoiCategoryName']} → {row['PoiCategoryName']} ({row['transition_count']})"
                for _, row in user_trans.iterrows()
            ]) or "No user transition data"

            # ========== 2. 获取该POI的邻近POI ==========
            poi_nearest = self.poi_nearest_df[self.poi_nearest_df['PoiId'].astype(int) == last_poi_id]
            nearest_text = ''
            if not poi_nearest.empty:
                nearest_text = f"{poi_nearest.iloc[0]['NearestPoiId'].astype(int)}"
            else:
                nearest_text = "No nearest POI data"

            # ========== 3. 获取该POI的高频转移目标 ==========
            poi_trans = self.poi_transition_df[self.poi_transition_df['PoiId'].astype(int) == last_poi_id]
            trans_text = ''
            if not poi_trans.empty:
                trans_text = f"{poi_trans.iloc[0]['NextPoiId'].astype(int)}"
            else:
                trans_text = "No POI transition data"

            # ========== 4. 构建语义检索文本 ==========
            retrieved_text = (
                f"User transitions: {user_trans_text}. "
                f"Top Near POIs: {nearest_text}. "
                f"Top transition POIs: {trans_text}. "
                f"Global category transitions: {all_trans_text}. "
            )

            # ========== 5. Sentence-BERT 编码 ==========
            retrieved_texts.append(retrieved_text)
    
        retrieved_vecs = self.sentence_encoder.encode(retrieved_texts, convert_to_tensor=True)
        self.retrieved_vecs.extend(retrieved_vecs)
        
        logging.warning("Batch processing of retrieved vectors completed.")


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], 
                # labels=self.labels[i],         
                answer_id = self.answer_ids[i],
                retrieved_vec=self.retrieved_vecs[i],  
                )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [instance.get("input_ids", None) for instance in instances]
        # 提取新的POI信息字段
        answer_ids = [instance.get("answer_id", None) for instance in instances]
        retrieved_vecs = [instance.get("retrieved_vec", None) for instance in instances]
        answers = torch.tensor(answer_ids, dtype=torch.long)
      
        
        retrieved_vecs = [torch.tensor(vec) if vec is not None else torch.zeros_like(retrieved_vecs[0]) for vec in retrieved_vecs]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            # labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            answers= answers,
            retrieved_vecs=torch.stack(retrieved_vecs),  
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, dataset=data_args.dataset, train=True)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, dataset=data_args.eval_dataset, train=False)  # 传入测试集路径
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

def evaluate(model, test_dataset, data_collator, device, batch_size=32):
    """
    用于评估模型在测试集上的表现，计算损失和准确率（ACC@1, ACC@5）。

    Parameters:
        model (nn.Module): 训练好的模型
        test_dataset (Dataset): 测试数据集
        data_collator (DataCollator): 数据合并器，用于处理批次数据
        device (torch.device): 使用的设备（cuda 或 cpu）
        batch_size (int, optional): 批次大小，默认为32

    Returns:
        dict: 包含损失、ACC@1、ACC@5等评估指标
    """
    # 切换到评估模式
    model.eval()
    
    # 准备测试数据加载器
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)
    
    # 初始化指标
    total_loss = 0
    correct_1 = 0
    correct_5 = 0
    total_samples = 0

    with torch.no_grad():  # 关闭梯度计算，提高效率
        for batch in tqdm(test_loader, desc="Evaluating"):
            # 将数据送到设备
            batch = {key: value.to(device) for key, value in batch.items()}

            # 获取输入数据和标签
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            # labels = batch['labels']
            answers = batch['answers']  # 假设这里是正确标签
            retrieval = batch['retrieved_vecs']

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # labels=labels,
                answers=answers,
                retrieval=retrieval
            )
            
            # 获取损失
            loss = outputs.get('loss', torch.tensor(0.0))
            logits = outputs.get('logits')

            # 累计损失
            total_loss += loss.item()

            # 获取前5个预测类别 (Top-5)
            _, preds = torch.topk(logits, 5, dim=1)

            # 计算 ACC@1 和 ACC@5
            correct_1 += (preds[:, 0] == answers).sum().item()  # ACC@1
            correct_5 += (preds == answers.unsqueeze(1)).sum().item()  # ACC@5
            total_samples += answers.size(0)

    # 计算最终评估结果
    avg_loss = total_loss / len(test_loader)
    acc_1 = correct_1 / total_samples  # ACC@1
    acc_5 = correct_5 / total_samples  # ACC@5

    # 返回评估结果
    return {
        "loss": avg_loss,
        "acc_1": acc_1,
        "acc_5": acc_5
    }
    
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    save_training_command(training_args.output_dir, os.path.basename(__file__),
                          model_args=model_args, data_args=data_args,
                          training_args=training_args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=4096,
        padding_side="right",
        use_fast=False,
    )

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        # _flash_attn_2_enabled = True,
    )

    # 动态扩展上下文窗口 修改位置编码的长度
    context_size = training_args.model_max_length if training_args.model_max_length > 0 else training_args.seq_len
    orig_ctx_len = getattr(config, "max_position_embeddings", None)  # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     llm_int8_threshold=6.0,
        #     llm_int8_has_fp16_weight=False,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # ),
    )
    model.resize_token_embeddings(151646)
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

    
    qwen_model = MoraModel.from_pretrained(model, training_args.output2_dir)
    peft_weights = torch.load(training_args.output2_dir + '/' + 'adapter_model.safetensors')
    qwen_model.load_state_dict(peft_weights, strict=False)
    
    model = ClassificationHead(config=model_args, model=qwen_model, sentence_bert_dim=768)
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    # 5. 训练过程
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # train_dataset = data_module['train_dataset']  # 获取训练数据集
    # data_collator = data_module['data_collator']  # 获取数据合并器

    #     # 早停机制变量初始化
    # best_loss = float('inf')  # 最佳损失初始化为无穷大
    # patience_counter = 0  # 没有改进的轮数计数器
    # for epoch in range(training_args.num_train_epochs):  # 使用 training_args 中的 num_train_epochs
    #     model.train()  # 设置为训练模式
    #     total_loss = 0
    #     correct_1 = 0  # 记录 ACC@1
    #     correct_5 = 0  # 记录 ACC@5
    #     total_samples = 0  # 总样本数
        
    #     # 遍历数据集
    #     for step in tqdm(range(0, len(train_dataset), training_args.batch_size), desc=f"Training epoch {epoch + 1}"):
    #         # 获取当前批次的数据（每次取 batch_size 个样本）
    #         batch = [train_dataset[i] for i in range(step, min(step + training_args.batch_size, len(train_dataset)))]
            
    #         # 使用数据合并器将数据拼接成一个批次
    #         batch = data_collator(batch)  # 返回处理后的批次数据
    #         # 将数据传送到设备
    #         input_ids = batch['input_ids'].to(device)
    #         attention_mask = batch['attention_mask'].to(device)
    #         # labels = batch['labels'].to(device)
            
    #         answers = batch['answers'].to(device) if 'answers' in batch else None

    #         retrieval = batch['retrieved_vecs'].to(device) if 'retrieved_vecs' in batch else None
            
    #         # 前向传播
    #         outputs = model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             # labels=labels,
    #             answers=answers,
    #             retrieval=retrieval
    #         )
            
    #         # 获取损失
    #         loss = outputs.get('loss')
    #         total_loss += loss.item()
            
    #         logits = outputs.get('logits')  # 预测的 logits
    #         _, preds = torch.topk(logits, 5, dim=1)  # 获取前 5 的预测类别 (Top-5)
            
    #         # 计算 ACC@1 和 ACC@5
    #         correct_1 += (preds[:, 0] == answers).sum().item()  # ACC@1
    #         correct_5 += (preds == answers.unsqueeze(1)).sum().item()  # ACC@5
    #         total_samples += answers.size(0)  # 统计样本总数

    #         # 反向传播与优化
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
        
    #     # 打印每个epoch的总损失
    #     avg_loss = total_loss / len(train_dataset)
    #     acc_1 = correct_1 / total_samples  # ACC@1
    #     acc_5 = correct_5 / total_samples  # ACC@5

    #     # 打印每个epoch的总损失和准确率
    #     print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, ACC@1: {acc_1:.4f}, ACC@5: {acc_5:.4f}")

    #     if avg_loss < best_loss:
    #         best_loss = avg_loss
    #         patience_counter = 0  # 重置没有改进的计数器
    #         torch.save(model.classifier.state_dict(), "output/umra_try/classifier.pth")

    #     else:
    #         patience_counter += 1
    #         if patience_counter >= training_args.early_stopping_patience:
    #             print(f"Early stopping at epoch {epoch + 1} due to no improvement in loss for {training_args.early_stopping_patience} epochs.")
    #             break  # 提前终止训练

    # 训练完成后，调用测试函数
    model.classifier.load_state_dict(torch.load("output/umra_try/classifier.pth"))
    model.eval()  # 切换到评估模式

    # 准备测试数据
    test_dataset = data_module['eval_dataset']
    data_collator = data_module['data_collator']

    # 调用 evaluate_model 函数
    eval_results = evaluate(model, test_dataset, data_collator, device, batch_size=training_args.batch_size)

    # 打印评估结果
    print(f"Test Loss: {eval_results['loss']:.4f}, ACC@1: {eval_results['acc_1']:.4f}, ACC@5: {eval_results['acc_5']:.4f}")


if __name__ == "__main__":
    train()
