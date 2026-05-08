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


class TrajectoryProjector(nn.Module):
    """将外部轨迹编码向量投影到路由空间"""
    def __init__(self, traj_dim: int, num_experts: int,
                 dtype=torch.bfloat16, dropout=0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(traj_dim, traj_dim // 2, dtype=dtype),
            nn.ReLU(),
            nn.Linear(traj_dim // 2, num_experts, dtype=dtype),
        )
        self._cached_traj_embedding: Optional[torch.Tensor] = None

    def set_trajectory_embedding(self, traj_embedding: torch.Tensor):
        self._cached_traj_embedding = traj_embedding

    def clear(self):
        self._cached_traj_embedding = None

    def forward(self, traj_embedding: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.proj(traj_embedding), dim=-1)


class TokenRouter(nn.Module):
    def __init__(self, config: UmRaConfig, input_dim: int,
                 layer_id: int, tag: Optional[str] = None,
                 use_trajectory: bool = False,
                 traj_projector: Optional[TrajectoryProjector] = None):
        super().__init__()
        alpha = -config.epsilon_alpha + 2 * config.epsilon_alpha * (
                layer_id / config.max_llm_layer) + config.alpha_shift
        alpha = torch.tensor(alpha, dtype=config.torch_dtype)
        if config.use_task_router:
            if config.task_router_only:
                self.task_router = TaskRouter(config)
                self.task_router_only = True
                self.alpha = None
            else:
                self.task_router_only = False
                if torch.sigmoid(alpha) < config.alpha_low_bound:
                    self.task_router = None
                    self.alpha = None
                elif torch.sigmoid(alpha) > config.alpha_up_bound:
                    self.task_router = TaskRouter(config)
                    self.alpha = None
                    self.task_router_only = True
                else:
                    self.alpha = nn.Parameter(alpha)
                    self.task_router = TaskRouter(config)
        else:
            self.task_router = None
            self.task_router_only = False
            self.alpha = None

        self.num_experts = config.num_experts
        self.layer_id = layer_id
        self.input_dim = input_dim
        self.torch_dtype = config.torch_dtype
        self.tag = tag
        self.dropout_prop = config.dropout
        self.top_k_routing_strategy = config.top_k_routing_strategy
        self.top_k = config.top_k
        self.trajectory_top_k_routing_strategy = getattr(
            config, "trajectory_top_k_routing_strategy", False
        )
        self.trajectory_top_k = getattr(config, "trajectory_top_k", self.top_k)

        self.use_trajectory = use_trajectory
        self.traj_projector = traj_projector
        self.active_top_k = self.num_experts

        if not self.task_router_only:
            if not self.use_trajectory:
                if config.num_router_mlp_layers == 1:
                    self.mlp = nn.Sequential(
                        nn.Dropout(self.dropout_prop),
                        nn.Linear(input_dim, self.num_experts,
                                  dtype=config.torch_dtype)
                    )
                else:
                    self.mlp = nn.Sequential(
                        nn.Dropout(self.dropout_prop),
                        nn.Linear(input_dim, config.router_hidden_dim,
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
            else:
                self.mlp = None
        else:
            self.mlp = None

        self.routing_weight: Optional[torch.Tensor] = None
        self.token_routing_weight: Optional[torch.Tensor] = None
        self.gamma_div_balance = config.gamma_div_balance_t
        self.gamma_div_certain = config.gamma_div_certain_t

    def forward(self, hidden_states: torch.Tensor):
        if self.task_router_only:
            routing_weight = self.task_router.get_task_weight().unsqueeze(-2)
            routing_weight = routing_weight.expand(
                hidden_states.shape[:-1] + (self.num_experts,))
            self.routing_weight = routing_weight
            return self.routing_weight

        if self.use_trajectory:
            traj_emb = None
            if self.traj_projector is not None:
                traj_emb = getattr(self.traj_projector, '_cached_traj_embedding', None)

            if traj_emb is not None:
                traj_emb = traj_emb.to(dtype=hidden_states.dtype,
                                        device=hidden_states.device)
                routing_weight = self.traj_projector(traj_emb)
            else:
                B = hidden_states.size(0)
                routing_weight = torch.ones(
                    B, self.num_experts,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device) / self.num_experts

            routing_weight = routing_weight.unsqueeze(1).expand(
                -1, hidden_states.size(1), -1)
        else:
            routing_input = hidden_states.mean(dim=1)
            routing_weight = F.softmax(self.mlp(routing_input), dim=-1)
            routing_weight = routing_weight.unsqueeze(1).expand(
                -1, hidden_states.size(1), -1)

        self.token_routing_weight = routing_weight

        if self.task_router is not None:
            task_weight = self.task_router.get_task_weight().unsqueeze(-2)
            alpha = torch.sigmoid(self.alpha)
            self.routing_weight = (1 - alpha) * routing_weight + alpha * task_weight
        else:
            self.routing_weight = routing_weight

        use_top_k = self.top_k_routing_strategy
        top_k = self.top_k
        if self.use_trajectory and self.trajectory_top_k_routing_strategy:
            use_top_k = True
            top_k = self.trajectory_top_k

        if use_top_k:
            top_k = max(1, min(top_k, self.num_experts))
            top_k_values, top_k_indices = torch.topk(
                self.routing_weight, top_k, dim=-1)
            rw = torch.full_like(self.routing_weight,
                                 torch.finfo(self.routing_weight.dtype).min)
            rw.scatter_(-1, top_k_indices, top_k_values)
            self.routing_weight = torch.softmax(rw, dim=-1)
            self.active_top_k = top_k
        else:
            self.active_top_k = self.num_experts

        return self.routing_weight

    def clear(self):
        if self.task_router is not None:
            self.task_router.clear()
        self.routing_weight = None
        self.token_routing_weight = None

    def get_routing_weight(self):
        return self.routing_weight

    def divergence_loss(self, attention_mask):
        if self.task_router_only:
            return torch.tensor(0, dtype=self.routing_weight.dtype, device=self.routing_weight.device)
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
        freq = torch.sum(count.view(-1, self.num_experts), dim=0) / (num_token * self.active_top_k)
        prop = torch.sum(routing_weight.view(-1, self.num_experts), dim=0) / num_token
        loss = torch.sum(prop * freq) * self.num_experts
        return loss.unsqueeze(0)


# ========== 修改点1：FusionGate 替换原来的 Router ==========
class FusionGate(nn.Module):
    """
    融合门控：决定 gate1(平均池化路由) 和 gate2(轨迹路由) 的权重
    输入：hidden_states平均池化 + 轨迹向量
    """
    def __init__(self, input_dim: int, traj_dim: int, 
                 dtype=torch.bfloat16, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim + traj_dim, (input_dim + traj_dim) // 2, dtype=dtype),
            nn.ReLU(),
            nn.Linear((input_dim + traj_dim) // 2, 2, dtype=dtype),
        )
        self.routing_weight: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor, 
                traj_embedding: torch.Tensor = None) -> torch.Tensor:
        """
        hidden_states: (B, T, D)
        traj_embedding: (B, traj_dim) 或 (traj_dim,)
        return: (B, 2) softmax权重
        """
        h = hidden_states.mean(dim=1)  # (B, D)
        
        if traj_embedding is not None:
            traj_embedding = traj_embedding.to(dtype=h.dtype, device=h.device)
            if traj_embedding.dim() == 1:
                traj_embedding = traj_embedding.unsqueeze(0)
            if traj_embedding.size(0) == 1 and h.size(0) > 1:
                traj_embedding = traj_embedding.expand(h.size(0), -1)
            
            h = torch.cat([h, traj_embedding], dim=-1)  # (B, D + traj_dim)
        
        self.routing_weight = F.softmax(self.gate(h), dim=-1)
        return self.routing_weight

    def clear(self):
        self.routing_weight = None

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
        self.router1_use_shared_expert = getattr(config, "router1_use_shared_expert", False)
        self.router1_shared_expert_weight = getattr(config, "router1_shared_expert_weight", 1.0)
        
        if self.use_hydra_lora:
            self.mora_a1 = nn.Parameter(torch.empty((self.rank, self.in_features), dtype=self.dtype_))
            self.mora_a2 = nn.Parameter(torch.empty((self.rank, self.in_features), dtype=self.dtype_))
        else:
            self.mora_a1 = nn.Parameter(torch.empty((self.rank * self.num_experts, self.in_features), dtype=self.dtype_))
            self.mora_a2 = nn.Parameter(torch.empty((self.rank * self.num_experts, self.in_features), dtype=self.dtype_))

        self.mora_b1 = nn.Parameter(torch.empty((self.out_features, self.rank * self.num_experts), dtype=self.dtype_))
        self.mora_b2 = nn.Parameter(torch.empty((self.out_features, self.rank * self.num_experts), dtype=self.dtype_))
        if self.router1_use_shared_expert:
            self.shared_mora_a1 = nn.Parameter(torch.empty((self.rank, self.in_features), dtype=self.dtype_))
            self.shared_mora_b1 = nn.Parameter(torch.empty((self.out_features, self.rank), dtype=self.dtype_))
        else:
            self.register_parameter("shared_mora_a1", None)
            self.register_parameter("shared_mora_b1", None)
        self.scaling = config.lora_alpha / math.sqrt(config.lora_r)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.mora_a1, a=math.sqrt(5))
        nn.init.zeros_(self.mora_b1)
        nn.init.kaiming_uniform_(self.mora_a2, a=math.sqrt(5))
        nn.init.zeros_(self.mora_b2)
        if self.router1_use_shared_expert:
            nn.init.kaiming_uniform_(self.shared_mora_a1, a=math.sqrt(5))
            nn.init.zeros_(self.shared_mora_b1)

    def forward(self, hidden_states: torch.Tensor,
                gate1: torch.Tensor,   # (B, T, E) 平均池化路由权重
                gate2: torch.Tensor,   # (B, T, E) 轨迹路由权重  
                fusion: torch.Tensor,  # (B, 2) 融合系数
                residual: torch.Tensor) -> torch.Tensor:

        x = self.dropout(hidden_states)

        # 专家组1（平均池化路由）
        h1 = F.linear(x, self.mora_a1)
        shape1 = h1.shape[:-1] + (self.num_experts, self.rank)
        h1 = h1.unsqueeze(-2).expand(shape1) if self.use_hydra_lora else h1.view(shape1)
        h1 = (h1 * gate1.unsqueeze(-1)).view(h1.shape[:-2] + (-1,))
        out1 = F.linear(h1, self.mora_b1) * self.scaling
        if self.router1_use_shared_expert:
            shared_h1 = F.linear(x, self.shared_mora_a1)
            shared_out1 = F.linear(shared_h1, self.shared_mora_b1) * self.scaling
            out1 = out1 + self.router1_shared_expert_weight * shared_out1

        # 专家组2（轨迹路由）
        h2 = F.linear(x, self.mora_a2)
        shape2 = h2.shape[:-1] + (self.num_experts, self.rank)
        h2 = h2.unsqueeze(-2).expand(shape2) if self.use_hydra_lora else h2.view(shape2)
        h2 = (h2 * gate2.unsqueeze(-1)).view(h2.shape[:-2] + (-1,))
        out2 = F.linear(h2, self.mora_b2) * self.scaling

        # 融合：w1*out1 + w2*out2
        w1 = fusion[:, 0].unsqueeze(-1).unsqueeze(-1)
        w2 = fusion[:, 1].unsqueeze(-1).unsqueeze(-1)
        output = w1 * out1 + w2 * out2

        return output.to(residual.dtype) + residual


class LoRA(nn.Module):
    def __init__(self, base_layer: nn.Linear, config: UmRaConfig):
        super().__init__()
        self.out_features, self.in_features = base_layer.weight.shape
        self.dtype_ = config.torch_dtype
        self.dropout_tate = config.dropout
        self.dropout = nn.Dropout(config.dropout)
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


# ========== 修改点2：AdapterLinear.forward 传递轨迹向量 ==========
class AdapterLinear(nn.Module):
    def __init__(self, base_layer: nn.Linear,
                 config: UmRaConfig,
                 router1: Optional[TokenRouter],
                 router2: Optional[TokenRouter],
                 fusion_gate: Optional[FusionGate],  # 改名：fusion_gate
                 use_cache: bool = False,
                 use_lora: bool = False):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.weight = base_layer.weight
        if hasattr(base_layer, "bias"):
            self.bias = base_layer.bias
        else:
            self.register_parameter('bias', None)

        self.use_lora = use_lora
        if self.use_lora:
            self.lora = LoRA(base_layer, config)
        else:
            self.mora = MoRa(base_layer, config)

        self.use_cache = use_cache
        self.router1 = router1
        self.router2 = router2
        self.fusion_gate = fusion_gate  # 存 FusionGate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        result = F.linear(hidden_states, self.weight, self.bias)

        if self.use_lora:
            return self.lora(hidden_states, result)

        if self.use_cache:
            gate1 = self.router1.get_routing_weight()
            gate2 = self.router2.get_routing_weight()
            fusion = self.fusion_gate_weight
        else:
            gate1 = self.router1(hidden_states)
            gate2 = self.router2(hidden_states)
            
            # ===== 获取轨迹向量传给 fusion_gate =====
            traj_emb = None
            if (self.router2 is not None 
                and getattr(self.router2, 'use_trajectory', False)
                and self.router2.traj_projector is not None):
                traj_emb = self.router2.traj_projector._cached_traj_embedding
            if self.router2 is not None and getattr(self.router2, "use_trajectory", False):
                if traj_emb is None:
                    raise RuntimeError(
                        f"Trajectory embedding missing in AdapterLinear (router2 tag={getattr(self.router2, 'tag', None)}). "
                        "This usually means projector cache was not set or was cleared before checkpoint recompute."
                    )
            fusion = self.fusion_gate(hidden_states, traj_emb)
            self.fusion_gate_weight = fusion

        return self.mora(hidden_states, gate1, gate2, fusion, result)


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
        task_embedding = self.task_embedding(torch.tensor([0], device=src.device))
        task_embedding = task_embedding.expand(src.shape[0], -1, -1)
        src = self.pos_encoder(src)
        src = torch.cat([src, task_embedding], dim=1)
        src = src.transpose(0, 1)
        src_key_padding_mask = torch.cat([src_key_padding_mask, torch.ones(src_key_padding_mask.shape[0], 1,
                                                                           device=src_key_padding_mask.device,
                                                                           dtype=src_key_padding_mask.dtype)], dim=1)
        src_key_padding_mask = src_key_padding_mask == 0
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        sentence_embedding = output[-1]
        return sentence_embedding


class MLPHead(nn.Module):
    def __init__(self, config, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc1 = self.fc1.to(dtype=config.torch_dtype)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.fc2 = self.fc2.to(dtype=config.torch_dtype)

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


# ========== 修改点3：_apply_for_layer 创建 FusionGate ==========
def _apply_for_layer(layer_module: nn.Module,
                     layer_id: int,
                     config: UmRaConfig,
                     traj_projector: Optional[TrajectoryProjector] = None):
    router_list = []      # 存 FusionGate
    token_router_list = []
    task_router_list = []

    def get_router(input_dim, tag: Optional[str] = None):
        # router1：平均池化路由
        router1 = TokenRouter(config, input_dim, layer_id,
                              tag=f"{tag}_G1",
                              use_trajectory=False,
                              traj_projector=None)
        # router2：轨迹向量路由
        router2 = TokenRouter(config, input_dim, layer_id,
                              tag=f"{tag}_G2",
                              use_trajectory=getattr(config, 'use_trajectory_routing', False),
                              traj_projector=traj_projector)
        
        # fusion gate：接收 hidden_states + traj_embedding
        traj_dim = getattr(config, 'trajectory_dim', input_dim)
        fusion_gate = FusionGate(input_dim, traj_dim,
                                 dtype=config.torch_dtype,
                                 dropout=config.dropout)

        token_router_list.append(router1)
        token_router_list.append(router2)
        router_list.append(fusion_gate)  # 存 FusionGate

        if router1.task_router is not None:
            task_router_list.append(router1.task_router)
        if router2.task_router is not None:
            task_router_list.append(router2.task_router)

        return router1, router2, fusion_gate

    def get_target_modules(target: list[str]) -> list[str]:
        res = []
        for t in target:
            res += TARGET_MODULE_TYPE[config.model_type][t]
        return res

    def set_mora(module, targets: list):
        use_cache = False
        for target_name in targets:
            if target_name not in config.target_modules:
                continue
            target_model = _get_module(module, target_name)
            if not isinstance(target_model, nn.Linear):
                continue
            if config.target_modules_lora is not None and target_name in config.target_modules_lora:
                target_model = AdapterLinear(target_model, config,
                                             router1=None, router2=None, fusion_gate=None,
                                             use_cache=False, use_lora=True)
            else:
                if not hasattr(set_mora, 'router1'):
                    set_mora.router1, set_mora.router2, set_mora.fusion_gate = get_router(
                        target_model.in_features, tag=target_name)
                target_model = AdapterLinear(target_model, config,
                                             router1=set_mora.router1, 
                                             router2=set_mora.router2, 
                                             fusion_gate=set_mora.fusion_gate,
                                             use_cache=use_cache, use_lora=False)
                use_cache = True
            setattr(module, target_name, target_model)

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

    return token_router_list, task_router_list, router_list


def _apply_hmora(model, config: UmRaConfig) -> PeftModel:
    def _extract_layer_id(name: str):
        match = re.search(r'\.\d+$', name)
        if match:
            return int(name.split('.')[-1])
        return None

    layer_list = dict()
    for module_name, module in model.named_modules():
        layer_id = _extract_layer_id(module_name)
        if layer_id is not None:
            if layer_id > config.max_llm_layer:
                config.max_llm_layer = layer_id
            layer_list[layer_id] = module

    traj_projector = None
    if getattr(config, 'use_trajectory_routing', False):
        traj_dim = getattr(config, 'trajectory_dim', config.hidden_size)
        share = getattr(config, 'share_traj_projector', True)
        if share:
            traj_projector = TrajectoryProjector(
                traj_dim=traj_dim,
                num_experts=config.num_experts,
                dtype=config.torch_dtype,
                dropout=config.dropout
            )
            print(f"  Created shared TrajectoryProjector: "
                  f"traj_dim={traj_dim} -> num_experts={config.num_experts}")

    token_router_list = nn.ModuleList()
    task_router_list = nn.ModuleList()
    router_list = nn.ModuleList()

    for layer_id in sorted(layer_list.keys()):
        module = layer_list[layer_id]

        layer_traj_proj = traj_projector
        if (getattr(config, 'use_trajectory_routing', False)
                and not getattr(config, 'share_traj_projector', True)):
            layer_traj_proj = TrajectoryProjector(
                traj_dim=getattr(config, 'trajectory_dim', config.hidden_size),
                num_experts=config.num_experts,
                dtype=config.torch_dtype,
                dropout=config.dropout
            )

        token_routers, task_routers, routers = _apply_for_layer(
            module, layer_id, config, traj_projector=layer_traj_proj)
        for router in token_routers:
            token_router_list.append(router)
        for router in task_routers:
            task_router_list.append(router)
        for router in routers:
            router_list.append(router)

    if config.use_task_router and len(task_router_list) > 0:
        embed = _get_module(model,
                            model.base_model_prefix + '.' +
                            TARGET_MODULE_TYPE[config.model_type]['embed'])
        task_embedding = embed.weight.data[config.task_token_id]
        task_encoder = TaskEncoder(config, task_embedding)
    else:
        task_encoder = None
    model.task_encoder = task_encoder

    router_manager = RouterManager(config, task_router_list, token_router_list, router_list)
    model.router_manager = router_manager

    if traj_projector is not None:
        model.traj_projector = traj_projector

    trainable_modules = ['router', 'router1', 'router2', 'mora', 'lora',
                         'task_encoder', 'traj_proj', 'fusion_gate']
    for param_name, param in model.named_parameters():
        if any(target in param_name for target in trainable_modules):
            param.requires_grad = True

    model.save_pretrained = types.MethodType(_save_pretrained, model)
    setattr(model, 'peft_config', config)
    return model


# ========== 修改点4：RouterManager.clear 清除 FusionGate ==========
class RouterManager(nn.Module):
    def __init__(self, config: UmRaConfig,
                 task_routers: nn.ModuleList,
                 token_routers: nn.ModuleList,
                 routers: nn.ModuleList):  # routers 现在存 FusionGate
        super().__init__()
        self.task_routers = task_routers
        self.token_routers = token_routers
        self.routers = routers
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
        for router in self.routers:  # FusionGate 也有 clear 方法
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
