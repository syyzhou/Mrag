# Written by Peibo Li
# Original code based on https://github.com/dvlab-research/LongLoRA?tab=readme-ov-file
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

import json
import os
import math
import pickle
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
import transformers
from peft import PeftModel
from model_lora import MoraModel
# from llama_attn_replace import replace_llama_attn
# from llama_attn_replace_sft import replace_llama_attn
from typing import Dict, Optional, Sequence
import sys
from transformers import BitsAndBytesConfig
import config

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size during inference')
    parser.add_argument('--base_model', type=str, default="./Qwen2-1.5B")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--seq_len', type=int, default=4096, help='context length during evaluation')
    parser.add_argument('--context_size', type=int, default=4096, help='context size during fine-tuning')
    parser.add_argument('--peft_model', type=str, default=None, help='')
    parser.add_argument('--flash_attn', type=bool, default=False, help='')
    parser.add_argument('--model_path', type=str, default='', help='your model path')
    parser.add_argument('--data_path', type=str, default="./test.bin", help='')
    parser.add_argument('--output_dir', type=str, default="./outputmodels/finetune-36/", help='')
    parser.add_argument('--dataset_name', type=str, default="nyc",
                        help='')
    parser.add_argument('--test_file', type=str, default="test_qa_pairs_kqt_100.txt",
                        help='')
    args = parser.parse_args()
    return args

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


def get_as_batch(data, seq_length, batch_size, device='cpu', sliding_window=256):
    all_ix = list(range(0, len(data) - seq_length, sliding_window))
    all_ix.pop() #list删除元素并返回该元素的具体值

    for idx in range(0, len(all_ix), batch_size):
        ix = all_ix[idx:idx + batch_size]
        assert all([idx + seq_length + 1 <= len(data) for idx in ix])
        x = torch.stack([torch.from_numpy((data[i:i + seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y


def iceildiv(x, y):
    return (x + y - 1) // y


def evaluate(model, data, batch_size, device, seq_length, sliding_window=256, use_cache=False):
    stats = {}

    model.eval()

    loss_list_val, acc_list = [], []
    loss_step_list_val = []

    with torch.no_grad():
        print(f"Using seq length {seq_length}")
        torch.set_printoptions(sci_mode=False)
        for idx, (x, y) in tqdm(
                enumerate(
                    get_as_batch(
                        data['val'],
                        seq_length,
                        batch_size,
                        device=device,
                        sliding_window=sliding_window
                    )
                ),
                total=iceildiv(
                    iceildiv(len(data['val']), sliding_window),
                    batch_size
                )
        ):
            val_loss = 0.
            acc = 0.
            cnt = 0

            for part_idx, i in enumerate(range(0, x.shape[1], seq_length)):
                part_len = x[:, i:i + seq_length].shape[1]

                outputs = model(
                    input_ids=x[:, i:i + seq_length],
                    labels=x[:, i:i + seq_length].contiguous(),
                    use_cache=use_cache)

                val_loss = outputs.loss * part_len + val_loss
                acc = ((outputs.logits.argmax(-1) == y[:, i:i + seq_length]).float().sum()) + acc
                cnt += part_len
                while len(loss_step_list_val) <= part_idx:
                    loss_step_list_val.append([])
                loss_step_list_val[part_idx].append(outputs.loss.item())
            val_loss /= cnt
            acc /= cnt

            loss_list_val.append(val_loss.item())
            acc_list.append(acc.item())

    stats['val_acc'] = torch.as_tensor(acc_list).mean().item()
    stats['val_loss'] = torch.as_tensor(loss_list_val).mean().item()
    stats['val_perplexity'] = 2.71828 ** stats['val_loss']
    stats['val_perplexity_per_chunk'] = torch.exp(torch.as_tensor(loss_step_list_val).mean(dim=1))

    return stats


def main(args):
    device = "cuda:7"
    seed = 2
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

    # print(tokenizer('6', return_tensors="pt").to(device))
    # print(tokenizer.decode([    29946]))
    # sys.exit()
    # if args.flash_attn:
    #     replace_llama_attn(inference=True)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_path,
        # _flash_attn_2_enabled = True,
    )

    # 动态扩展上下文窗口 修改位置编码的长度
    context_size = args.context_size if args.context_size > 0 else args.seq_len
    orig_ctx_len = getattr(config, "max_position_embeddings", None)  # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
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
    
    def load_lora_config_pickle(path_pickle):
        with open(path_pickle, 'rb') as pickle_file:
            lora_config_obj = pickle.load(pickle_file)
        return lora_config_obj

    if output_dir:    
        # lora_config = load_lora_config_pickle(os.path.join(output_dir, "lora_config.pkl"))  # 加载 LoraConfig 对象
            # 2. 加载LoRA配置
        # lora_config = torch.load(os.path.join(output_dir, "lora_config.pkl"))

        model = MoraModel.from_pretrained(model, output_dir)
        
     
        # def get_lora_params_dict(model):
        #     return {name: param.data.clone() for name, param in model.named_parameters() if 'lora' in name.lower()}
        
        # def get_mora_params_dict(model):
        #     return {name: param.data.clone() for name, param in model.named_parameters() if 'mora' in name.lower()}

        # def check_update(before, after, step_name):
        #     updated = sum(1 for name in before if name in after and not torch.allclose(before[name], after[name], atol=1e-6))
        #     print(f"{step_name}: {updated}个参数更新")
        #     return updated > 0

        # # 验证第一个权重加载
        # print("验证第一个权重加载...")
        # before_first = get_mora_params_dict(model)
        peft_weights = torch.load(output_dir + '/' + 'adapter_model.safetensors')

        # 加载权重到模型
        model.load_state_dict(peft_weights, strict=False)

        model = PeftModel.from_pretrained(model, os.path.join(output_dir, "Desc_lora"))

        
        # model.load_state_dict(peft_weights, strict=False)
        # after_first = get_mora_params_dict(model)
        # check_update(before_first, after_first, "adapter_model权重")

        # # 验证第二个权重加载
        # print("验证第二个权重加载...")
        # before_second = get_lora_params_dict(model)
        # model_weights_path = os.path.join(output_dir, "layer27_lora")
        # model = PeftModel.from_pretrained(model, model_weights_path)
        
        # lora_params = [name for name, param in model.named_parameters() if 'lora' in name]
        # print(f"✅ 找到 {len(lora_params)} 个LoRA参数")
        
        # # 检查一些LoRA参数
        # for name in lora_params[:3]:
        #     param = dict(model.named_parameters())[name]
        #     print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")
    

        # after_second = get_lora_params_dict(model)
        # check_update(before_second, after_second, "adapter_lora权重")
        
        # peft_weights = torch.load(output_dir + '/' + 'adapter_model.safetensors')
        # model.load_state_dict(peft_weights, strict=False)
        
        # model_weights_path = os.path.join(output_dir, "adapter_lora.safetensors")
        # model.load_state_dict(torch.load(model_weights_path), strict=False)
        
    model.eval()
    model.to(device)

    # 特殊token检查
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
    generation_config = transformers.GenerationConfig(
        max_new_tokens=30,
        min_new_tokens=None,
        # Generation strategy
        do_sample=False,
        num_beams=5,
        # num_beam_groups=5,
        # penalty_alpha=None,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,

        # Hyperparameters for logit manipulation
        # temperature=0.6,
        # top_k=40,
        # top_p=0.1,
        # typical_p=1.0,
        # diversity_penalty=4.0,
        repetition_penalty=1.176,
        # length_penalty=1.0,
        # no_repeat_ngram_size=0,

        # num_return_sequences=1 
        num_return_sequences=5
    )
    import re
    def evaluate_prediction_accuracy(prediction, ground_truth):
        # Regular expression to extract POI ids from prediction and ground truth
        pred_poi_pattern1 = r"POI id (\d+)."
        # pred_poi_pattern1 = r'is: (\d+)'
        pred_poi_pattern2 = r"(\d+)."
        # pred_poi_pattern = r"with POI id (\d+)."
        # pred_poi_pattern = r"visited POI id (\d+) with Category Name"
        # pred_poi_pattern = r'will visit POI ([^\.]+)\.'
        # pred_poi_pattern = r'will visit POI ([^\.]+) which is'
        # Extract predicted and actual POI ids
        if "POI id" in prediction:
            predicted_poi = re.search(pred_poi_pattern1, prediction).group(1)
        elif "." in prediction:
            predicted_poi = prediction[:-1]
        else:
            predicted_poi = prediction
        actual_poi = re.search(pred_poi_pattern1, ground_truth).group(1)
        # predicted_poi = prediction[:-1]

        # Compare and return accuracy (1 if they match, 0 otherwise)
        return int(predicted_poi == actual_poi)

    data_path = f'./datasets/{args.dataset_name}/preprocessed/'
    with open(data_path + f"{args.test_file}", "r") as file:
        lines = file.readlines()
    correct_predictions_1 = 0
    correct_predictions_5 = 0
    correct_predictions_10 = 0
    model.eval()
    # device = 'cuda:0'
    # Iterate over each line and ask the LLM
    correct_list = []
    for index, line in tqdm(enumerate(lines), desc="Processing lines", total=len(lines)):
        prompt1, gt = line.split("<answer>:")
        # prompt, gt = line.split("<answer>:")
        tmp1, tmp2 = prompt1.split('Which POI id will user ')
        time = tmp1[-24:]
        # txt1, res = gt.split("is:")
        # user_id = re.search(r'"user_id":\s*(\d+)', prompt1).group(1)
        # time = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', gt).group(0)
        # tmp1, tmp2 = prompt1.split('will visit at ')
        # user_id = tmp1.split('user_')[-1]
        user_id = tmp2.split(' visit')[0]
        prompt = prompt1.replace('<question>:', '<question>:') + '<answer>:' + f'{time}, user {user_id} will visit POI id '
        # 提取时间：从tmp2的前19个字符中获取时间
        # time = tmp2[:19]
        
        # time = tmp1[-24:]
        # user_id = tmp2.split(' visit')[0]
        # prompt = prompt1.replace('<question>:', '<question>:') + '<answer>:' + f'{time}, user {user_id} will visit POI id '
        # prompt = prompt1.replace('<question>:', '<question>:') + '<answer>:' + f'The next POI ID that user_{user_id} will visit at {time} is : '
        # prompt = prompt1 + txt1 + f'is :'
        # prompt1, prompt2, gt = line.split("<answer>:")
        # prompt = prompt1.replace('<question>:', '<user>:\n') + "\n<assistant>:\n"
        # if len(tokenizer.tokenize(prompt)) >= 32768:
        if len(tokenizer.tokenize(prompt)) >= 4096:
            continue
        # prompt = tokenizer(prompt, return_tensors="pt").to(device)       
        with torch.no_grad():
            if model.task_encoder is not None:
                prefix_tensors = tokenizer(prompt, padding=True, return_tensors='pt', add_special_tokens=False).to(device)
                embedding = getattr(model.base_model, "embed_tokens")
                hidden_states = embedding(prefix_tensors["input_ids"])
                task_embed = model.task_encoder(hidden_states, prefix_tensors["attention_mask"])
                model.router_manager.set_task_weight(task_embed)
        prompt = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**prompt, generation_config=generation_config)
        # prediction = tokenizer.decode(outputs[:, prompt.input_ids.shape[1]:][0], skip_special_tokens=True).replace('[',
        #                                                                                                            '').replace(
        #     ']', '')
        gt = gt.replace('[', '').replace(']', '')
        # gt = re.search(r"'next_poi_id': \[(\d+)\]", gt).group(1) 
        i = 0
        seen_predictions = set()
        while i < 5:
            try:
                prediction = tokenizer.decode(outputs[:, prompt.input_ids.shape[1]:][i],
                                              skip_special_tokens=True)
                # prediction = re.sub(r'[^0-9]', '', prediction)
                prediction = re.match(r'^\d+', prediction).group(0)
                i += 1
                # print(prediction)
                # print(gt)
                if prediction in seen_predictions:
                    continue
                seen_predictions.add(prediction)
                tmp = evaluate_prediction_accuracy(prediction, gt)
                if tmp:
                    if i == 1:
                        correct_list.append(index)
                        correct_predictions_1 += tmp
                        correct_predictions_5 += tmp
                        # correct_predictions_10 += tmp
                        # break
                    elif i < 6:
                        correct_predictions_5 += tmp
                        # correct_predictions_10 += tmp
                        break
                    else:
                        # correct_predictions_10 += tmp
                        break
            except:
                continue
        # sys.exit()

    print(f'ACC@1:{correct_predictions_1 / len(lines)}')
    print(f'ACC@5:{correct_predictions_5 / len(lines)}')
    print(f'ACC@10:{correct_predictions_10 / len(lines)}')
    print(f'correct_index:{correct_list}')


if __name__ == "__main__":
    args = parse_config()
    main(args)

