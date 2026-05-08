import json
import os
import re
import types
from typing import Optional

import math
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import PreTrainedModel
import seaborn as sns

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
        print(f"config:{config}")
        model = _apply_hmora(model, config=config)
        return model

class Time2Vec(nn.Module):
    """
    Time2Vec: 可学习的时间表示
    论文: https://arxiv.org/abs/1907.05321
    
    t2v(τ)[i] = ωi * τ + φi        (i = 0, 线性项)
    t2v(τ)[i] = sin(ωi * τ + φi)   (i > 0, 周期项)
    """
    def __init__(self, input_dim: int, embed_dim: int, dtype=torch.bfloat16):
        super().__init__()
        self.input_dim = input_dim  # 时间特征数量（如 hour, weekday, is_weekend = 3）
        self.embed_dim = embed_dim  # 每个时间特征的嵌入维度
        
        # 可学习的频率 ω 和相位 φ
        # 对每个输入特征，学习 embed_dim 个周期分量
        self.omega = nn.Parameter(torch.randn(input_dim, embed_dim, dtype=dtype) * 0.1)
        self.phi = nn.Parameter(torch.zeros(input_dim, embed_dim, dtype=dtype))
        
        # 线性项的权重和偏置
        self.linear_weight = nn.Parameter(torch.randn(input_dim, 1, dtype=dtype) * 0.1)
        self.linear_bias = nn.Parameter(torch.zeros(input_dim, 1, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] - 归一化的时间特征
               例如: [hour/24, weekday/7, is_weekend]
        Returns:
            [B, input_dim * (embed_dim + 1)] - 时间嵌入向量
        """
        # x: [B, input_dim]
        batch_size = x.shape[0]
        
        # 线性项: [B, input_dim, 1]
        linear = x.unsqueeze(-1) * self.linear_weight + self.linear_bias
        
        # 周期项: [B, input_dim, embed_dim]
        # ω*τ + φ
        periodic_input = x.unsqueeze(-1) * self.omega + self.phi  # [B, input_dim, embed_dim]
        periodic = torch.sin(periodic_input)
        
        # 拼接: [B, input_dim, embed_dim + 1]
        time_embed = torch.cat([linear, periodic], dim=-1)
        
        # 展平: [B, input_dim * (embed_dim + 1)]
        time_embed = time_embed.view(batch_size, -1)
        
        return time_embed

class TimeEncoder(nn.Module):
    """
    时间编码器：使用 Time2Vec + 融合层
    """
    def __init__(self, config: UmRaConfig):
        super().__init__()
        self.torch_dtype = config.torch_dtype
        self.time_embed_dim = config.time_embed_dim
        
        # 输入: 3个时间特征 (hour, weekday, is_weekend)
        self.num_time_features = 3
        self.time2vec_dim = 16  # 每个特征的 Time2Vec 周期维度
        
        # Time2Vec 编码器
        self.time2vec = Time2Vec(
            input_dim=self.num_time_features,
            embed_dim=self.time2vec_dim,
            dtype=config.torch_dtype
        )
        
        # Time2Vec 输出维度: num_time_features * (time2vec_dim + 1)
        t2v_output_dim = self.num_time_features * (self.time2vec_dim + 1)
        
        # 融合 MLP
        self.fusion = nn.Sequential(
            nn.Linear(t2v_output_dim, config.time_embed_dim, dtype=config.torch_dtype),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.time_embed_dim, config.time_embed_dim, dtype=config.torch_dtype)
        )
        
    def forward(self, time_info: dict) -> torch.Tensor:
        """
        Args:
            time_info: dict containing:
                - 'hour': torch.Tensor [B], 值域 0-23
                - 'weekday': torch.Tensor [B], 值域 0-6 (0=Monday)
                - 'is_weekend': torch.Tensor [B], 值域 0 or 1
        
        Returns:
            time_vector: torch.Tensor [B, time_embed_dim]
        """
        # 提取并归一化时间特征
        hour = time_info['hour'].float() / 24.0  # [B]
        weekday = time_info['weekday'].float() / 7.0  # [B]
        is_weekend = time_info['is_weekend'].float()  # [B], 已经是 0/1
        
        # 堆叠成 [B, 3]
        time_features = torch.stack([hour, weekday, is_weekend], dim=-1).to(self.torch_dtype)
        
        # Time2Vec 编码
        time_embed = self.time2vec(time_features)  # [B, 3 * (16 + 1)] = [B, 51]
        
        # 融合层
        time_vector = self.fusion(time_embed)  # [B, time_embed_dim]
        
        return time_vector
    
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
                 layer_id: int, tag: Optional[str] = None, time_aware: bool = False):
        super().__init__()
        
        self.time_aware = time_aware
        self.time_embed_dim = config.time_embed_dim if time_aware else 0
        # 路由输入维度 = hidden_size + time_embed_dim (如果启用时间感知)
        input_dim = input_dim + self.time_embed_dim
       
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

    def forward(self, hidden_states: torch.Tensor, time_vector: torch.Tensor = None):
        # if self.mlp is not None:
        #      # 获取 MLP 第一层的权重类型
        #     target_dtype = self.mlp[1].weight.dtype if isinstance(self.mlp[0], nn.Dropout) else self.mlp[0].weight.dtype
        #     hidden_states = hidden_states.to(target_dtype)
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
             # ========== 拼接时间向量 ==========
            # if self.time_aware and time_vector is not None:
            #     routing_input = torch.cat([routing_input, time_vector], dim=-1)  # [B, D + time_dim]
                
            if self.time_aware and time_vector is not None:
                # 处理 beam search 导致的 batch size 不匹配
                if time_vector.size(0) != routing_input.size(0):
                    time_vector = time_vector.expand(routing_input.size(0), -1)
                routing_input = torch.cat([routing_input, time_vector], dim=-1)  # [B, D + time_dim]
                
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
                 routers: nn.ModuleList,
                 adapter_linears: nn.ModuleList = None):
        super().__init__()
        self.task_routers = task_routers
        self.token_routers = token_routers
        self.routers = routers
        self.adapter_linears = adapter_linears if adapter_linears is not None else nn.ModuleList()
        # loss
        self.top_k_routing_strategy = config.top_k_routing_strategy
        self.use_load_balancing_loss = config.use_load_balancing_loss
        self.use_div_loss = config.use_div_loss

        self.lambda_auxiliary = config.lambda_auxiliary
        self.lambda_lm = config.lambda_lm

    # def set_time_vector(self, time_vector: torch.Tensor):
    #     """为所有 AdapterLinear 设置时间向量"""
    #     for adapter in self.adapter_linears:
    #         adapter.set_time_vector(time_vector)
            
    def set_time_vector(self, time_vector: torch.Tensor):
        """为所有 AdapterLinear 设置时间向量"""
        # ✅ 存储原始 time_vector 用于梯度计算
        self._time_vector = time_vector
        # ✅ 传递 detach 版本给 adapter（避免重复计算图）
        detached_time_vector = time_vector.detach()
        for adapter in self.adapter_linears:
            adapter.set_time_vector(detached_time_vector)

    def set_task_weight(self, task_embedding: torch.Tensor):
        for task_router in self.task_routers:
            task_router(task_embedding)

    def clear(self):
        for router in self.token_routers:
            router.clear()
        for router in self.routers:
            router.clear()
        # 清除时间向量
        for adapter in self.adapter_linears:
            adapter.clear_time_vector()

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
        # print(f"hidden:{type(hidden_states)}")
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
    def __init__(self, base_layer: nn.Linear,
                 config: UmRaConfig,
                 router1: Optional[TokenRouter],
                 router2: Optional[TokenRouter],
                 router: Optional[Router],
                 use_cache: bool = False,
                 use_lora: bool = False):
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

        self.use_lora = use_lora
        # mixture of lora experts
        if self.use_lora:
            self.lora = LoRA(base_layer, config)
        else:
            self.mora = MoRa(base_layer, config)
        # routers
        self.use_cache = use_cache # for router sharing
        self.router1 = router1
        self.router2 = router2
        self.router = router
        # ========== 新增：时间向量缓存 ==========
        self._cached_time_vector: Optional[torch.Tensor] = None
        # self.dropout_prop = config.dropout
        # self.gate_mlp = nn.Sequential(
        #             nn.Dropout(self.dropout_prop),
        #             nn.Linear(self.in_features, self.num_feats,
        #                       dtype=config.torch_dtype)
        #         )
        # self.category_proj = nn.Linear(self.in_features, self.in_features, dtype=config.torch_dtype)
        # self.id_proj = nn.Linear(self.in_features, self.in_features, dtype=config.torch_dtype)
        self.attention_scores = []
        self.max_sequence_count = 10 
        self.plot_called = False  #

    def set_time_vector(self, time_vector: torch.Tensor):
        """设置当前 batch 的时间向量"""
        self._cached_time_vector = time_vector
    
    def clear_time_vector(self):
        """清除缓存的时间向量"""
        self._cached_time_vector = None
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_lora:
            result = F.linear(hidden_states, self.weight, self.bias)
            return self.lora(hidden_states, result)
        else:
            if self.use_cache:
                # sharing router, the gate values has been calculated
                gate = self.router.get_routing_weight()
            else:
                # calculate gate values
                # category_feat = torch.tanh(self.category_proj(hidden_states))
                # id_feat = torch.tanh(self.id_proj(hidden_states))
                # gate1 = self.router1(hidden_states)
                gate1 = self.router1(hidden_states, time_vector=self._cached_time_vector)
                gate2 = self.router2(hidden_states)
                gate = self.router(hidden_states)
                # routing_input = hidden_states.mean(dim=1)  # shape: [B, D]
                # gate = F.softmax(self.gate_mlp(routing_input), dim=-1)  # shape: [B, num_experts]
                # self.attention_scores.append((gate1, gate2, gate))
                # # 当保存了10个序列时，自动调用可视化函数
                #   # 检查条件是否满足，并且只调用一次 plot_attention_heatmap()
                # if len(self.attention_scores) >= self.max_sequence_count and not self.plot_called:
                #     self.plot_attention_heatmap()
                #     self.plot_called = True  # 设置标志，确保只调用一次

                
            # print(f"Gate dtype: {gate.dtype}") 
            # print(f"hidden_states dtype: {hidden_states.dtype}") 
            result = F.linear(hidden_states, self.weight, self.bias)
            return self.mora(hidden_states, hidden_states, gate1, gate2, gate, result)
        
    def plot_attention_heatmap(self):
        """
        从模型中提取并可视化当前保存的注意力得分。
        """
        all_gate1 = []
        all_gate2 = []
        all_gate = []
        
        # 获取注意力得分（gate1, gate2, gate）
        for gate1, gate2, gate in self.attention_scores:
            gate1_reshaped = gate1[0, 0, :].reshape(1, 4)
            gate2_reshaped = gate2[0, 0, :].reshape(1, 4)
            gate_reshaped = gate[0, :].reshape(1, 2)

            all_gate1.append(gate1_reshaped.to(torch.float32).cpu().numpy())
            all_gate2.append(gate2_reshaped.to(torch.float32).cpu().numpy())
            all_gate.append(gate_reshaped.to(torch.float32).cpu().numpy())
        
        # 转换为 NumPy 数组
        all_gate1 = torch.tensor(all_gate1)  # shape: [num_sequences, num_experts_per_pool]
        all_gate2 = torch.tensor(all_gate2)  # shape: [num_sequences, num_experts_per_pool]
        all_gate = torch.tensor(all_gate)    # shape: [num_sequences, num_experts_per_pool]

        # 合并 gate1, gate2，形成一个三维张量 (num_sequences, num_expert_pools, num_experts_per_pool)
        attention_scores = torch.stack([all_gate1, all_gate2], dim=1)  # shape: [num_sequenc
        
        # 可视化堆叠条形图：展示每个专家池的注意力得分
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        colors = ['#A3C4BC', '#F9E79F', '#F6B4B1', '#F19C42']  # 不同专家的颜色
        # 转换数据并去除不必要的维度
        attention_scores_np = attention_scores.numpy().squeeze()  # 去除大小为1的维度
        # print(f"处理后数据形状: {attention_scores_np.shape}")  # 应该是 (10, 2, 4)

        for i in range(2):  # 2个专家池
            ax = axes[i]
            
            # 获取当前专家池的所有专家数据
            pool_scores = attention_scores_np[:, i, :]  # 形状应该是 (10, 4)
            # print(f"专家池 {i+1} 数据形状: {pool_scores.shape}")
            
            # 初始化底部数据
            bottom = np.zeros(pool_scores.shape[0])
            
            for j in range(pool_scores.shape[1]):  # 每个专家 (4个专家)
                # 获取当前专家的得分，确保是一维数组
                expert_scores = pool_scores[:, j]  # 形状应该是 (10,)
                # print(f"专家 {j+1} 得分形状: {expert_scores.shape}")
                
                # 绘制水平条形图
                ax.barh(range(len(expert_scores)), expert_scores, 
                        left=bottom, color=colors[j], label=f'expert {j+1}')
                
                # 更新底部数据
                bottom += expert_scores

            ax.set_title(f'expert pool {i+1}')
            ax.set_xlabel('score')
            ax.set_ylabel('seq')
            ax.set_yticks(range(pool_scores.shape[0]))
            ax.set_yticklabels([f'seq {k+1}' for k in range(pool_scores.shape[0])])
            ax.set_xlim(0, 1.0)
            ax.legend(title="expert", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig('attention_scores_stack_plot.png', bbox_inches='tight', dpi=300)
        plt.show()
        
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
    def __init__(self, config, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc1 = self.fc1.to(dtype=config.torch_dtype)  # 将dtype应用到层上
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.fc2 = self.fc2.to(dtype=config.torch_dtype)  # 将dtype应用到层上


    def forward(self, x):
        x = x.mean(dim=1)  
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
    adapter_linear_list = []  # 新增

    def get_router(input_dim, tag: Optional[str] = None):
        # router = TokenRouter(config, input_dim, layer_id, tag=tag)
        # token_router_list.append(router)
        # if router.task_router is not None:
        #     task_router_list.append(router.task_router)
        # return router
        # 创建两个 TokenRouter 实例
        router1 = TokenRouter(config, input_dim, layer_id, tag=tag, time_aware=config.use_time_aware_routing)  # 时间感知
        router2 = TokenRouter(config, input_dim, layer_id, tag=tag, time_aware=False)
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
            if config.target_modules_lora is not None and target_name in config.target_modules_lora:
                target_model = AdapterLinear(target_model, config, router=None, use_cache=False, use_lora=True)
            else:
                if token_router is None:
                    # token_router = get_router(target_model.in_features, tag=target_name)
                    router1, router2, router = get_router(target_model.in_features, tag=target_name)
                target_model = AdapterLinear(target_model, config, router1=router1, router2=router2, router=router, use_cache=use_cache,
                                             use_lora=False)
                adapter_linear_list.append(target_model)  # 收集 adapter
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

    return token_router_list, task_router_list, router_list, adapter_linear_list


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

    token_router_list = nn.ModuleList()
    task_router_list = nn.ModuleList()
    router_list = nn.ModuleList()
    adapter_linear_list = nn.ModuleList()  # 新增
    # apply for each layer
    for layer_id in sorted(layer_list.keys()):
        module = layer_list[layer_id]
        token_routers, task_routers, routers, adapters = _apply_for_layer(module, layer_id, config)
        for router in token_routers:
            token_router_list.append(router)
        for router in task_routers:
            task_router_list.append(router)
        for router in routers:
            router_list.append(router)
        for adapter in adapters:
            adapter_linear_list.append(adapter)
    
     # ========== 新增：创建时间编码器 ==========
    if config.use_time_aware_routing:
        time_encoder = TimeEncoder(config)
    else:
        time_encoder = None
    model.time_encoder = time_encoder
    # ==========================================
    
    if config.use_task_router and len(task_router_list) > 0:
        embed = _get_module(model, model.base_model_prefix + '.' + TARGET_MODULE_TYPE[config.model_type]['embed'])
        task_embedding = embed.weight.data[config.task_token_id]
        task_encoder = TaskEncoder(config, task_embedding)
    else:
        task_encoder = None
    model.task_encoder = task_encoder
    # router manager
    router_manager = RouterManager(config, task_router_list, token_router_list, router_list, adapter_linear_list)
    model.router_manager = router_manager
    # 替换llm-head 
    # mlp_head = MLPHead(config, config.hidden_size, config.hidden_size, config.poi_num)
    # if hasattr(model, "lm_head"):
    #     setattr(model, "lm_head", mlp_head)

    trainable_modules = ['router', 'router1', 'router2', 'mora', 'lora', 'task_encoder', 'time_encoder', 'time2vec', 'fusion']
    # trainable_modules = []
    # trainable_modules = ['router', 'mora']
    # freeze parameters
    for param_name, param in model.named_parameters():
        if any(target in param_name for target in trainable_modules):
            param.requires_grad = True
        # else:
        #     param.requires_grad = False

    # overwrite save_pretrained
    model.save_pretrained = types.MethodType(_save_pretrained, model)
    # model.peft_config = config
    setattr(model, 'peft_config', config)
    return model


def _save_pretrained(self: nn.Module, path):
    if not os.path.exists(path):
        os.makedirs(path)
    trainable_params = dict()
    for name, param in self.named_parameters():
        if param.requires_grad:
            trainable_params[name] = param.detach().cpu()
    config = self.peft_config.export()
    torch.save(trainable_params, path + '/' + 'adapter_model.safetensors')
    config['torch_dtype'] = None
    json.dump(config, open(path + '/' + 'config.json', 'w'))
