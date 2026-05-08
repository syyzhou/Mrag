from dataclasses import asdict
import json
import os
import pickle
import re
import types
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel, LoraConfig, get_peft_model as peft_get_peft_model
from transformers import PreTrainedModel

from config import (
    UmRaConfig,
    TARGET_MODULE_TYPE
)


class MoraModel:
    @staticmethod
    def from_pretrained(
            model: PreTrainedModel,
            name_or_path: Optional[str] = None,
    ) -> PeftModel:
        with open(name_or_path + "/config.json") as f:
            config = json.load(f)
        config = UmRaConfig.from_config(config)
        config.torch_dtype = model.dtype
        model = _apply_hmora(model, config=config)
        return model


class TaskRouter(nn.Module):
    def __init__(self, config: UmRaConfig):
        super().__init__()
        self.torch_dtype = config.torch_dtype
        self.num_experts: int = config.num_experts
        self.dropout_prop = config.dropout
        if config.num_router_mlp_layers == 1:
            self.mlp = nn.Sequential(
                nn.Dropout(self.dropout_prop),
                nn.Linear(config.hidden_size, self.num_experts, dtype=config.torch_dtype)
            )
        else:
            self.mlp = nn.Sequential(nn.Dropout(self.dropout_prop),
                                     nn.Linear(config.hidden_size, config.router_hidden_dim,
                                               dtype=config.torch_dtype),
                                     nn.ReLU())
            for i in range(config.num_router_mlp_layers - 2):
                self.mlp.append(nn.Dropout(self.dropout_prop))
                self.mlp.append(nn.Linear(config.router_hidden_dim, config.router_hidden_dim,
                                          dtype=config.torch_dtype))
                self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(self.dropout_prop))
            self.mlp.append(nn.Linear(config.router_hidden_dim, self.num_experts,
                                      dtype=config.torch_dtype))
        self.gamma_div_balance = config.gamma_div_balance_s
        self.gamma_div_certain = config.gamma_div_certain_s

        self.task_weight: Optional[torch.Tensor] = None

    def forward(self, task_presentation: torch.Tensor):
        self.task_weight = F.softmax(self.mlp(task_presentation), dim=-1)

    def divergence_loss(self):
        # generalized Jensen-Shannon (GJS) divergence loss
        task_weight_batched = self.task_weight
        max_entropy = torch.log(torch.tensor(
            self.num_experts, dtype=task_weight_batched.dtype, device=task_weight_batched.device))
        max_entropy_m = self.gamma_div_balance * max_entropy
        min_entropy_p = self.gamma_div_certain * max_entropy
        max_div = max_entropy_m - min_entropy_p
        m = torch.mean(task_weight_batched, dim=0)
        hm = -torch.sum(m * torch.log(m + 1e-9), dim=-1)
        hm = torch.clamp(hm, max=max_entropy_m)
        hp = -torch.sum(task_weight_batched * torch.log(task_weight_batched + 1e-9), dim=-1)
        hp = torch.clamp(hp, min=min_entropy_p)
        hp = torch.mean(hp, dim=-1)
        loss = torch.relu(max_div - (hm - hp)) / max_entropy
        return loss

    def get_task_weight(self):
        return self.task_weight

    def clear(self):
        self.task_weight = None


class TokenRouter(nn.Module):
    def __init__(self, config: UmRaConfig, input_dim: int,
                 layer_id: int, tag: Optional[str] = None):
        super().__init__()
        # alpha
        alpha = -config.epsilon_alpha + 2 * config.epsilon_alpha * (
                layer_id / config.max_llm_layer) + config.alpha_shift
        alpha = torch.tensor(alpha, dtype=config.torch_dtype)
        if config.use_task_router:
            # use task router
            if config.task_router_only:
                self.task_router = TaskRouter(config)
                self.task_router_only = True
                self.alpha = None
            else:
                self.task_router_only = False
                if torch.sigmoid(alpha) < config.alpha_low_bound:
                    # token router only
                    self.task_router = None
                    self.alpha = None
                elif torch.sigmoid(alpha) > config.alpha_up_bound:
                    # task router only
                    self.task_router = TaskRouter(config)
                    self.alpha = None
                    self.task_router_only = True
                else:
                    # combine token and task router
                    self.alpha = nn.Parameter(alpha)
                    self.task_router = TaskRouter(config)
        else:
            # token router only
            self.task_router = None
            self.task_router_only = False
            self.alpha = None

        self.num_experts = config.num_experts
        self.layer_id = layer_id
        self.input_dim = input_dim
        self.torch_dtype = config.torch_dtype
        self.tag = tag

        self.dropout_prop = config.dropout
        # routing strategy
        self.top_k_routing_strategy = config.top_k_routing_strategy
        self.top_k = config.top_k
        # routing function
        if not self.task_router_only:
            # token router parameters
            if config.num_router_mlp_layers == 1:
                self.mlp = nn.Sequential(
                    nn.Dropout(self.dropout_prop),
                    nn.Linear(input_dim, self.num_experts,
                              dtype=config.torch_dtype)
                )
            else:
                # mlp input
                self.mlp = nn.Sequential(nn.Dropout(self.dropout_prop),
                                         nn.Linear(input_dim, config.router_hidden_dim,
                                                   dtype=config.torch_dtype),
                                         nn.ReLU())
                # mlp hidden
                for i in range(config.num_router_mlp_layers - 2):
                    self.mlp.append(nn.Dropout(self.dropout_prop))
                    self.mlp.append(nn.Linear(config.router_hidden_dim, config.router_hidden_dim,
                                              dtype=config.torch_dtype))
                    self.mlp.append(nn.ReLU())
                # mlp output
                self.mlp.append(nn.Dropout(self.dropout_prop))
                self.mlp.append(nn.Linear(config.router_hidden_dim, self.num_experts,
                                          dtype=config.torch_dtype))
        else:
            self.mlp = None
            
        # gamma
        self.routing_weight: Optional[torch.Tensor] = None
        self.token_routing_weight: Optional[torch.Tensor] = None
        self.gamma_div_balance = config.gamma_div_balance_t
        self.gamma_div_certain = config.gamma_div_certain_t

    def forward(self, hidden_states: torch.Tensor):
        if self.task_router_only:
            routing_weight = self.task_router.get_task_weight().unsqueeze(-2)
            routing_weight = routing_weight.expand(hidden_states.shape[:-1] + (self.num_experts,))
            self.routing_weight = routing_weight
        else:
            # 原始分token
            # routing_weight = F.softmax(self.mlp(hidden_states), dim=-1)
            # self.token_routing_weight = routing_weight
            
            # 所有平均池化
            routing_input = hidden_states.mean(dim=1)  # shape: [B, D]
            routing_weight = F.softmax(self.mlp(routing_input), dim=-1)  # shape: [B, num_experts]
            routing_weight = routing_weight.unsqueeze(1).expand(-1, hidden_states.size(1), -1)  # shape: [B, T, num_experts]

            # 最后一个token
            # routing_input = hidden_states[:, -1, :]             # shape: [B, D]
            # routing_weight = F.softmax(self.mlp(routing_input), dim=-1)  # shape: [B, num_experts]
            # routing_weight = routing_weight.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
            self.token_routing_weight = routing_weight
            if self.task_router is not None:
                task_weight = self.task_router.get_task_weight().unsqueeze(-2)
                alpha = torch.sigmoid(self.alpha)
                self.routing_weight = (1 - alpha) * routing_weight + alpha * task_weight
            else:
                self.routing_weight = routing_weight
        if self.top_k_routing_strategy:
            top_k_values, top_k_indices = torch.topk(self.routing_weight, self.top_k, dim=-1)
            routing_weight = torch.full_like(self.routing_weight, torch.finfo(self.routing_weight.dtype).min)
            routing_weight.scatter_(-1, top_k_indices, top_k_values)
            routing_weight = torch.softmax(routing_weight, dim=-1)
            self.routing_weight = routing_weight
        return self.routing_weight

    def get_routing_weight(self):
        return self.routing_weight

    def divergence_loss(self, attention_mask):
        if self.task_router_only:
            return torch.tensor(0, dtype=self.routing_weight.dtype, device=self.routing_weight.device)
        # generalized Jensen-Shannon (GJS) divergence loss
        token_routing_weight = self.token_routing_weight
        mask = attention_mask.to(token_routing_weight.dtype).unsqueeze(-1)
        token_routing_weight = token_routing_weight * mask
        max_entropy = torch.log(torch.tensor(
            self.num_experts, dtype=token_routing_weight.dtype, device=token_routing_weight.device))
        max_entropy_m = self.gamma_div_balance * max_entropy
        min_entropy_p = self.gamma_div_certain * max_entropy
        max_div = max_entropy_m - min_entropy_p
        num_token = torch.sum(mask)
        token_routing_weight = token_routing_weight * mask
        m = torch.sum(token_routing_weight.view(-1, self.num_experts), dim=0) / num_token
        entropy_m = -torch.sum(m * torch.log(m + 1e-9), dim=-1)
        entropy_m = torch.clamp(entropy_m, max=max_entropy_m)
        entropy_p = -torch.sum(token_routing_weight * torch.log(token_routing_weight + 1e-9), dim=-1)
        entropy_p = torch.clamp(entropy_p, min=min_entropy_p) * mask.squeeze(-1)
        entropy_p = torch.sum(entropy_p) / num_token
        loss = torch.relu(max_div - (entropy_m - entropy_p)) / max_entropy
        return loss

    def load_balancing_loss(self, attention_mask):
        routing_weight = self.token_routing_weight
        mask = attention_mask.to(routing_weight.dtype)
        num_token = mask.sum()
        routing_weight = routing_weight * mask.unsqueeze(-1)
        count = torch.sign(self.routing_weight * mask.unsqueeze(-1))
        freq = torch.sum(count.view(-1, self.num_experts), dim=0) / (num_token * self.top_k)
        prop = torch.sum(routing_weight.view(-1, self.num_experts), dim=0) / num_token
        loss = torch.sum(prop * freq) * self.num_experts
        return loss.unsqueeze(0)

    def clear(self):
        if self.task_router is not None:
            self.task_router.clear()
        self.routing_weight = None
        self.token_routing_weight = None


class RouterManager(nn.Module):
    def __init__(self, config: UmRaConfig,
                 task_routers: nn.ModuleList,
                 token_routers: nn.ModuleList,
                 routers: nn.ModuleList):
        super().__init__()
        self.task_routers = task_routers
        self.token_routers = token_routers
        self.routers = routers
        # loss
        self.top_k_routing_strategy = config.top_k_routing_strategy
        self.use_load_balancing_loss = config.use_load_balancing_loss
        self.use_div_loss = config.use_div_loss

        self.lambda_auxiliary = config.lambda_auxiliary
        self.lambda_lm = config.lambda_lm

    def set_task_weight(self, task_embedding: torch.Tensor):
        for task_router in self.task_routers:
            task_router(task_embedding)

    def clear(self):
        for router in self.token_routers:
            router.clear()
        for router in self.routers:
            router.clear()

    def backward_auxiliary_loss_for_seq_router(self, reduce='sum'):
        if self.use_load_balancing_loss:
            return 0

        auxiliary_loss = []
        for router in self.task_routers:
            if self.use_div_loss:
                auxiliary_loss.append(router.divergence_loss())
            else:
                break

        if len(auxiliary_loss) == 0:
            return 0

        loss = torch.stack(auxiliary_loss, dim=0)
        if reduce == 'sum':
            loss = torch.sum(loss)
        elif reduce == 'mean':
            loss = torch.mean(loss)
        else:
            raise ValueError(f'reduce must be sum or mean, but got {reduce}')
        loss = loss * self.lambda_auxiliary
        loss.backward()
        return loss.item()

    def get_auxiliary_loss(self, loss, attention_mask, reduce='sum'):
        auxiliary_loss = []
        for router in self.token_routers:
            if self.use_load_balancing_loss:
                auxiliary_loss.append(router.load_balancing_loss(attention_mask))
            elif self.use_div_loss:
                auxiliary_loss.append(router.divergence_loss(attention_mask))
            else:
                break
        if len(auxiliary_loss) == 0:
            return loss
        auxiliary_loss = torch.stack(auxiliary_loss, dim=0)
        if reduce == 'sum':
            auxiliary_loss = torch.sum(auxiliary_loss)
        elif reduce == 'mean':
            auxiliary_loss = torch.mean(auxiliary_loss)
        else:
            raise ValueError(f'reduce must be sum or mean, but got {reduce}')
        loss = self.lambda_lm * loss + self.lambda_auxiliary * auxiliary_loss
        return loss

class Router(nn.Module):
    def __init__(self, config: UmRaConfig, input_dim: int,
                 layer_id: int, tag: Optional[str] = None):
        super().__init__()
        # alpha
        self.num_experts = config.num_experts
        self.layer_id = layer_id
        self.input_dim = input_dim
        self.torch_dtype = config.torch_dtype
        self.tag = tag
        self.num_feats = 2
        
        self.dropout_prop = config.dropout
        self.gate_mlp = nn.Sequential(
                    nn.Dropout(self.dropout_prop),
                    nn.Linear(self.input_dim, self.num_feats,
                              dtype=config.torch_dtype)
                )
            
        # gamma
        self.routing_weight: Optional[torch.Tensor] = None
        self.token_routing_weight: Optional[torch.Tensor] = None
        
    def forward(self, hidden_states: torch.Tensor):
        # 原始分token
        # routing_weight = F.softmax(self.mlp(hidden_states), dim=-1)
        # self.token_routing_weight = routing_weight
        
        # 所有平均池化
        routing_input = hidden_states.mean(dim=1)  # shape: [B, D]
        routing_weight = F.softmax(self.gate_mlp(routing_input), dim=-1)  # shape: [B, num_experts]
        # routing_weight = routing_weight.unsqueeze(1).expand(-1, hidden_states.size(1), -1)  # shape: [B, T, num_experts]

        # 最后一个token
        # routing_input = hidden_states[:, -1, :]             # shape: [B, D]
        # routing_weight = F.softmax(self.mlp(routing_input), dim=-1)  # shape: [B, num_experts]
        # routing_weight = routing_weight.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        self.gate_routing_weight = routing_weight
        self.routing_weight = routing_weight
       
        return self.routing_weight
    
    def clear(self):
        self.routing_weight = None
        self.gate_routing_weight = None
    
    def get_routing_weight(self):
        return self.routing_weight
        
class MoRa(nn.Module):
    def __init__(self, base_layer: nn.Linear, config: UmRaConfig):
        super().__init__()
        self.out_features, self.in_features = base_layer.weight.shape
        self.dtype_ = config.torch_dtype

        self.num_experts = config.num_experts
        self.rank = config.lora_r

        self.dropout_tate = config.dropout
        self.dropout = nn.Dropout(p=config.dropout)
        self.use_hydra_lora = config.use_hydra_lora
        
        if self.use_hydra_lora:
            self.mora_a1 = nn.Parameter(torch.empty((self.rank, self.in_features), dtype=self.dtype_))
            self.mora_a2 = nn.Parameter(torch.empty((self.rank, self.in_features), dtype=self.dtype_))
        else:
            self.mora_a1 = nn.Parameter(torch.empty((self.rank * self.num_experts, self.in_features), dtype=self.dtype_))
            self.mora_a2 = nn.Parameter(torch.empty((self.rank * self.num_experts, self.in_features), dtype=self.dtype_))

        self.mora_b1 = nn.Parameter(torch.empty((self.out_features, self.rank * self.num_experts), dtype=self.dtype_))
        self.mora_b2 = nn.Parameter(torch.empty((self.out_features, self.rank * self.num_experts), dtype=self.dtype_))
        # rs_lora scaling
        self.scaling = config.lora_alpha / math.sqrt(config.lora_r)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.mora_a1, a=math.sqrt(5))
        nn.init.zeros_(self.mora_b1)
        nn.init.kaiming_uniform_(self.mora_a2, a=math.sqrt(5))
        nn.init.zeros_(self.mora_b2)


    def forward(self, hidden_states1: torch.Tensor, hidden_states2: torch.Tensor, gate1: torch.Tensor, gate2: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states1 = self.dropout(hidden_states1)
         # --- Group 1 ---
          # 组1
        h1 = F.linear(hidden_states1, self.mora_a1)  # [B, T, rank] 或 [B, T, rank * num_experts]
        target_shape_1 = h1.shape[:-1] + (self.num_experts, self.rank)  
        if self.use_hydra_lora:
            h1 = h1.unsqueeze(-2).expand(target_shape_1)  # Hydra 扩展形状
        else:
            h1 = h1.view(target_shape_1)                  # 普通多专家 reshape
        h1 = (h1 * gate1.unsqueeze(-1)).view(h1.shape[:-2] + (-1,))  # 加权融合 + 拉平
        out1 = F.linear(h1, self.mora_b1) * self.scaling             # 线性变换输出 [B,T,out_dim]

        # 组2
        h2 = F.linear(hidden_states2, self.mora_a2)
        target_shape_2 = h2.shape[:-1] + (self.num_experts, self.rank)
        if self.use_hydra_lora:
            h2 = h2.unsqueeze(-2).expand(target_shape_2)
        else:
            h2 = h2.view(target_shape_2)
        h2 = (h2 * gate2.unsqueeze(-1)).view(h2.shape[:-2] + (-1,))
        out2 = F.linear(h2, self.mora_b2) * self.scaling

         # 两组加权融合
        # weights = gate
        # output = weights[:, 0].unsqueeze(-1) * out1 + weights[:, 0].unsqueeze(-1) * out2
        output = (gate[:, 0].unsqueeze(-1).unsqueeze(-1) * out1) + (gate[:, 1].unsqueeze(-1).unsqueeze(-1) * out2)
        
        return output.to(residual.dtype) + residual

class LoRA(nn.Module):
    def __init__(self, base_layer: nn.Linear, config: UmRaConfig):
        super().__init__()
        self.out_features, self.in_features = base_layer.weight.shape
        self.dtype_ = config.torch_dtype
        self.dropout_tate = config.dropout
        self.dropout = nn.Dropout(config.dropout)
        # rs_lora scaling
        self.scaling = config.lora_alpha / math.sqrt(config.lora_r)
        self.rank = config.lora_r
        self.lora_a = nn.Parameter(
            torch.empty((self.rank, self.in_features), dtype=self.dtype_))
        self.lora_b = nn.Parameter(
            torch.empty((self.out_features, self.rank), dtype=self.dtype_))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = F.linear(hidden_states, self.lora_a)
        hidden_states = F.linear(hidden_states, self.lora_b) * self.scaling
        return hidden_states + residual


class AdapterLinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, base_name,
                 config: UmRaConfig,
                 router1: Optional[TokenRouter],
                 router2: Optional[TokenRouter],
                 router: Optional[Router],
                 use_cache: bool = False,
                 use_lora: bool = False,
                 ):
        super().__init__()
        # linear
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.weight = base_layer.weight
        self.num_feats=2
        
        if hasattr(base_layer, "bias"):
            self.bias = base_layer.bias
        else:
            self.register_parameter('bias', None)
        # print("Base Layer Structure:")
        # print(base_layer)

        self.use_lora = use_lora
        # self.is_last_layer = is_last_layer  # 是否为最后一层
        # mixture of lora experts
        # if self.is_last_layer:
        #     # 只在 MoRa 后加 LoRA 微调层
        #     lora_config = LoraConfig(
        #         r=8,  # LoRA的低秩矩阵秩
        #         lora_alpha=16,  # LoRA alpha 参数
        #         target_modules=['Linear'],  # 只对权重进行LoRA微调
        #         lora_dropout=0.1  # LoRA的dropout
        #     )
        #     # 使用 `get_peft_model` 将 LoRA 应用到最后一层
        #     self.lora = peft_get_peft_model(base_layer, lora_config)
        # else:
        #     self.mora = MoRa(base_layer, config)
        
        self.lora = base_layer
        self.mora = MoRa(base_layer, config)
        # routers
        self.use_cache = use_cache # for router sharing
        self.router1 = router1
        self.router2 = router2
        self.router = router
        # self.dropout_prop = config.dropout
        # self.gate_mlp = nn.Sequential(
        #             nn.Dropout(self.dropout_prop),
        #             nn.Linear(self.in_features, self.num_feats,
        #                       dtype=config.torch_dtype)
        #         )
        # self.category_proj = nn.Linear(self.in_features, self.in_features, dtype=config.torch_dtype)
        # self.id_proj = nn.Linear(self.in_features, self.in_features, dtype=config.torch_dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_lora:
            result = F.linear(hidden_states, self.weight, self.bias)
            return self.lora(hidden_states, result)
        if self.use_cache:
            # sharing router, the gate values has been calculated
            gate = self.router.get_routing_weight()
        else:
            # calculate gate values
            # category_feat = torch.tanh(self.category_proj(hidden_states))
            # id_feat = torch.tanh(self.id_proj(hidden_states))
            gate1 = self.router1(hidden_states)
            gate2 = self.router2(hidden_states)
            gate = self.router(hidden_states)
            # routing_input = hidden_states.mean(dim=1)  # shape: [B, D]
            # gate = F.softmax(self.gate_mlp(routing_input), dim=-1)  # shape: [B, num_experts]
            
        # print(f"Gate dtype: {gate.dtype}") 
        # print(f"hidden_states dtype: {hidden_states.dtype}") 
        # result = F.linear(hidden_states, self.weight, self.bias)
        # if self.is_last_layer:
        result = self.lora(hidden_states)
        # else: result = F.linear(hidden_states, self.weight, self.bias)
        return self.mora(hidden_states, hidden_states, gate1, gate2, gate, result)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.bfloat16).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        # for left padding position encoding of task encoder
        return x + self.pe[:, -x.size(1):, :]


class TaskEncoder(nn.Module):
    def __init__(self, config: UmRaConfig, task_embedding: Optional[torch.Tensor]):
        super(TaskEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_encoder_layer = config.num_encoder_layer
        self.pos_encoder = PositionalEncoding(self.hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(self.hidden_size, nhead=16,
                                                    dim_feedforward=self.hidden_size * 2, dropout=config.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.num_encoder_layer)
        self.task_embedding = torch.nn.Embedding(1, config.hidden_size, dtype=config.torch_dtype)
        if task_embedding is not None:
            self.task_embedding.weight.data.copy_(task_embedding)

    def forward(self, src, src_key_padding_mask):
        # src shape: (batch_size, seq_len, hidden_size)
        # src_key_padding_mask shape: (batch_size, seq_len)

        task_embedding = self.task_embedding(torch.tensor([0], device=src.device))
        task_embedding = task_embedding.expand(src.shape[0], -1, -1)
        src = self.pos_encoder(src)
        src = torch.cat([src, task_embedding], dim=1)
        src = src.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
        src_key_padding_mask = torch.cat([src_key_padding_mask, torch.ones(src_key_padding_mask.shape[0], 1,
                                                                           device=src_key_padding_mask.device,
                                                                           dtype=src_key_padding_mask.dtype)], dim=1)
        # attention_mask = self.gen_causal_attention_mask(hidden_states, padding_mask)
        src_key_padding_mask = src_key_padding_mask == 0
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        sentence_embedding = output[-1]
        # take the output of the last token as task presentation
        return sentence_embedding

class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def get_peft_model(model: PreTrainedModel, config: UmRaConfig) -> PeftModel:
    config.hidden_size = model.config.hidden_size
    config.model_type = model.config.model_type
    if model.config.model_type == "bloom":
        config.share_router_for_qkv = True
    model = _apply_hmora(model, config)
    return model


def _get_module(model: nn.Module, target_name: str):
    for name, module in model.named_modules():
        if name == target_name:
            return module
    return None


def _apply_for_layer(layer_module: nn.Module,
                     layer_id: int,
                     config: UmRaConfig):
    router_list = []
    token_router_list = []
    task_router_list = []

    def get_router(input_dim, tag: Optional[str] = None):
        # router = TokenRouter(config, input_dim, layer_id, tag=tag)
        # token_router_list.append(router)
        # if router.task_router is not None:
        #     task_router_list.append(router.task_router)
        # return router
        # 创建两个 TokenRouter 实例
        router1 = TokenRouter(config, input_dim, layer_id, tag=tag)
        router2 = TokenRouter(config, input_dim, layer_id, tag=tag)
        router = Router(config, input_dim, layer_id, tag=tag)
        # 将它们分别添加到 token_router_list
        token_router_list.append(router1)
        token_router_list.append(router2)
        router_list.append(router)
        
        # 如果存在 task_router，创建两个实例并添加到 task_router_list
        if router1.task_router is not None:
            task_router_list.append(router1.task_router)
            task_router_list.append(router2.task_router)
        return router1, router2, router

    def get_target_modules(target: list[str]) -> list[str]:
        res = []
        for t in target:
            res += TARGET_MODULE_TYPE[config.model_type][t]
        return res

    def set_mora(module, targets: list):
        """
        applay hmora for a list of linear, share the same router
        :param module:
        :param targets:
        :return:
        """
        token_router = None
        use_cache = False
        for target_name in targets:
            if target_name not in config.target_modules:
                continue
            target_model = _get_module(module, target_name)
            if not isinstance(target_model, nn.Linear):
                continue
            # is_last_layer = (layer_id == 27)
            if config.target_modules_lora is not None and target_name in config.target_modules_lora:
                target_model = AdapterLinear(target_model, target_name, config, router=None, use_cache=False, use_lora=True)
            else:
                if token_router is None:
                    # token_router = get_router(target_model.in_features, tag=target_name)
                    router1, router2, router = get_router(target_model.in_features, tag=target_name)
                target_model = AdapterLinear(target_model, target_name, config, router1=router1, router2=router2, router=router, use_cache=use_cache,
                                             use_lora=False)
                use_cache = True  # besides the first linear, others use the cached routing weight
            setattr(module, target_name, target_model)

    # apply for attention block
    atte_name = TARGET_MODULE_TYPE[config.model_type]['atte']
    atte_module = _get_module(layer_module, atte_name)
    if config.share_router_for_qkv:
        target_modules = get_target_modules(['q', 'k', 'v'])
        set_mora(atte_module, target_modules)
        target_modules = get_target_modules(['o'])
        for target_module in target_modules:
            set_mora(atte_module, [target_module])
    else:
        target_modules = get_target_modules(['q', 'k', 'v', 'o'])
        for target_module in target_modules:
            set_mora(atte_module, [target_module])

    # # apply for ffn block
    # ffn_name = TARGET_MODULE_TYPE[config.model_type]['ffn']
    # ffn_module = _get_module(layer_module, ffn_name)
    # if config.share_router_for_w_i:
    #     target_modules = get_target_modules(['wi'])
    #     set_mora(ffn_module, target_modules)
    #     target_modules = get_target_modules(['wo'])
    #     for target_module in target_modules:
    #         set_mora(ffn_module, [target_module])
    # else:
    #     target_modules = get_target_modules(['wi', 'wo'])
    #     for target_module in target_modules:
    #         set_mora(ffn_module, [target_module])

    return token_router_list, task_router_list, router_list


def _apply_hmora(model, config: UmRaConfig) -> PeftModel:
    """
    inject hmora into base model
    :param model: pretrain model
    :param config: hmora config
    :return: peft model
    """

    def _extract_layer_id(name: str):
        """
        extract layer id from module name
        :param name:
        :return:
        """
        # modules in module list end with digit
        match = re.search(r'\.\d+$', name)
        if match:
            return int(name.split('.')[-1])
        return None

    layer_list = dict()  # {layer_id : layer_module}
    for module_name, module in model.named_modules():
        layer_id = _extract_layer_id(module_name)
        if layer_id is not None:
            if layer_id > config.max_llm_layer:
                # record the max layer id
                config.max_llm_layer = layer_id
            # record decoder layers
            layer_list[layer_id] = module
    # lora_config = LoraConfig(
    #     r=8,  # LoRA的低秩矩阵秩
    #     lora_alpha=16,  # LoRA alpha 参数
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 微调最后一层的q_proj, k_proj, v_proj, o_proj
    #     lora_dropout=0,  # LoRA的dropout
    #     bias="none",  # 不微调bias
    #     task_type="CAUSAL_LM"  # 任务类型
    # )
    # last_layer = model.model.layers[-1]  # 假设最后一层是 Qwen2DecoderLayer
    # last_layer_model = peft_get_peft_model(last_layer, lora_config, adapter_name="last_layer_lora")  
     
    token_router_list = nn.ModuleList()
    task_router_list = nn.ModuleList()
    router_list = nn.ModuleList()
    # apply for each layer
    for layer_id in sorted(layer_list.keys()):
        # if layer_id == config.max_llm_layer:
        #     continue  # 跳过最后一层
        module = layer_list[layer_id]
        token_routers, task_routers, routers= _apply_for_layer(module, layer_id, config)
        for router in token_routers:
            token_router_list.append(router)
        for router in task_routers:
            task_router_list.append(router)
        for router in routers:
            router_list.append(router)
    
        # 处理最后一层的特殊情况
    # last_layer_module = layer_list[last_layer_id]
    # # 可以在这里对最后一层进行单独处理
    # # 例如：为最后一层添加特定的 LoRA 微调、MoRa 配置等
    # token_routers, task_routers, routers = _apply_for_layer(last_layer_module, last_layer_id, config)

    # 将最后一层的路由器添加到列表
    token_router_list.extend(token_routers)
    task_router_list.extend(task_routers)
    router_list.extend(routers)

    
    if config.use_task_router and len(task_router_list) > 0:
        embed = _get_module(model, model.base_model_prefix + '.' + TARGET_MODULE_TYPE[config.model_type]['embed'])
        task_embedding = embed.weight.data[config.task_token_id]
        task_encoder = TaskEncoder(config, task_embedding)
    else:
        task_encoder = None
    model.task_encoder = task_encoder
    # router manager
    router_manager = RouterManager(config, task_router_list, token_router_list, router_list)
    model.router_manager = router_manager
    # 替换llm-head 
    # mlp_head = MLPHead(config.hidden_size, config.hidden_size, config.poi_num)
    # if hasattr(model, "lm_head"):
    #     setattr(model, "lm_head", mlp_head)

    # trainable_modules = ['router', 'router1', 'router2', 'mora', 'lora', 'task_encoder']
    trainable_modules = ['lora_A', 'lora_B']
    # freeze parameters
    for param_name, param in model.named_parameters():
        if any(target in param_name for target in trainable_modules):
            param.requires_grad = True
        else:
            param.requires_grad = False

 
    # overwrite save_pretrained
    model.save_pretrained = types.MethodType(_save_pretrained, model)
    # model.peft_config = config
    setattr(model, 'peft_config', config)
    return model


# def _save_pretrained(self: nn.Module, path):
#     if not os.path.exists(path):
#         os.makedirs(path)
#     trainable_params = dict()
#     for name, param in self.named_parameters():
#         if param.requires_grad:
#             trainable_params[name] = param.detach().cpu()
#     # config = self.peft_config.export()
#     config = self.peft_config
#     torch.save(trainable_params, path + '/' + 'adapter_model.safetensors')
#     config['torch_dtype'] = None
#     json.dump(config, open(path + '/' + 'config.json', 'w'))

def save_lora_config_pickle(lora_config_obj, path_pickle):
    with open(path_pickle, 'wb') as pickle_file:
        pickle.dump(lora_config_obj, pickle_file)

# 2. 提取其他基础配置并保存为 JSON
def save_other_config_json(config_dict, path_json):
    # 提取 `LoraConfig` 部分，保留其余配置
    config_dict_copy = config_dict.copy()
    config_dict_copy.pop('Desc_lora', None) 
    with open(path_json, 'w') as json_file:
        json.dump(config_dict_copy, json_file, indent=4)


def _save_pretrained(self: nn.Module, path):
    if not os.path.exists(path):
        os.makedirs(path)
    trainable_params = dict()
    for name, param in self.named_parameters():
        if param.requires_grad:
            trainable_params[name] = param.detach().cpu()
    
    # LoraConfig 对象有 to_dict() 方法
    config = self.peft_config
    config['torch_dtype'] = None
    # 保存 LoraConfig 部分（Pickle）和其他配置部分（JSON）
    save_lora_config_pickle(config['Desc_lora'], os.path.join(path, 'Desc_lora/lora_config.pkl'))  # 保存 LoraConfig 对象
    save_other_config_json(config, os.path.join(path, 'config.json'))  # 保存基础配置

    
    torch.save(trainable_params, os.path.join(path, 'Desc_lora/adapter_lora.safetensors'))
