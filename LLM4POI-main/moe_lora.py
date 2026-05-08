import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class MoELoRAConfig:
    """MoE-LoRA 配置"""
    # LoRA 基础配置
    lora_r: int = 32  # 总的 rank，必须能被 num_experts 整除
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    # MoE 配置
    num_experts: int = 4  # 专家数量
    top_k: int = 2  # 激活的专家数量
    
    # Router 配置
    router_hidden_dim: int = 256
    router_type: str = "token"  # "token" or "sequence"
    use_load_balancing_loss: bool = True
    load_balance_weight: float = 0.01
    
    # 目标模块
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # 数据类型
    torch_dtype: str = "bfloat16"
    
    def __post_init__(self):
        assert self.lora_r % self.num_experts == 0, \
            f"lora_r ({self.lora_r}) must be divisible by num_experts ({self.num_experts})"
        self.expert_rank = self.lora_r // self.num_experts
class ExpertRouter(nn.Module):
    """
    专家路由器：决定每个 token/sequence 使用哪些专家
    """
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        router_type: str = "token",  # "token" or "sequence"
        hidden_dim: int = 256,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_type = router_type
        self.dtype = dtype
        
        # 路由网络
        self.router = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts, dtype=dtype)
        )
        
        # 用于计算负载均衡损失
        self._routing_weights: Optional[torch.Tensor] = None
        self._expert_usage: Optional[torch.Tensor] = None
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        return_all_weights: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [B, T, D] 或 [B, D]
            return_all_weights: 是否返回所有专家的权重（用于软混合）
            
        Returns:
            routing_weights: [B, T, num_experts] 或 [B, num_experts]
            expert_indices: [B, T, top_k] 或 [B, top_k]
        """
        # 根据路由类型处理输入
        if self.router_type == "sequence":
            # 序列级路由：使用平均池化
            if hidden_states.dim() == 3:
                router_input = hidden_states.mean(dim=1)  # [B, D]
            else:
                router_input = hidden_states
        else:
            # Token 级路由
            router_input = hidden_states
        
        # 计算路由 logits
        router_logits = self.router(router_input)  # [B, T, E] 或 [B, E]
        
        # 计算 softmax 权重
        routing_weights = F.softmax(router_logits, dim=-1)
        self._routing_weights = routing_weights
        
        # Top-K 选择
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        
        # 重新归一化 top-k 权重
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 计算专家使用率（用于负载均衡损失）
        self._compute_expert_usage(routing_weights, top_k_indices)
        
        if return_all_weights:
            # 创建稀疏权重矩阵
            sparse_weights = torch.zeros_like(routing_weights)
            sparse_weights.scatter_(-1, top_k_indices, top_k_weights)
            return sparse_weights, top_k_indices
        
        return top_k_weights, top_k_indices
    
    def _compute_expert_usage(
        self, 
        routing_weights: torch.Tensor, 
        top_k_indices: torch.Tensor
    ):
        """计算专家使用统计"""
        # 统计每个专家被选中的次数
        expert_mask = F.one_hot(top_k_indices, self.num_experts).float()
        expert_mask = expert_mask.sum(dim=-2)  # 跨 top_k 求和
        
        if routing_weights.dim() == 3:
            expert_mask = expert_mask.sum(dim=1)  # 跨 token 求和
        
        self._expert_usage = expert_mask.mean(dim=0)  # 跨 batch 求平均
    
    def load_balancing_loss(self) -> torch.Tensor:
        """
        计算负载均衡损失，鼓励专家均匀使用
        """
        if self._routing_weights is None:
            return torch.tensor(0.0)
        
        routing_weights = self._routing_weights
        
        # 计算每个专家的平均权重
        if routing_weights.dim() == 3:
            mean_weights = routing_weights.mean(dim=[0, 1])  # [E]
        else:
            mean_weights = routing_weights.mean(dim=0)  # [E]
        
        # 计算专家使用频率
        expert_freq = self._expert_usage / self._expert_usage.sum()
        
        # 负载均衡损失 = 专家数 * Σ(频率 * 平均权重)
        loss = self.num_experts * (expert_freq * mean_weights).sum()
        
        return loss
    
    def clear(self):
        """清除缓存"""
        self._routing_weights = None
        self._expert_usage = None



class MoELoRALayer(nn.Module):
    """
    基于 Rank 拆分的 MoE-LoRA 层
    
    将 LoRA 的 A 和 B 矩阵按 rank 维度拆分成多个专家：
    - A = [A₁, A₂, ..., Aₖ]，每个 Aₖ ∈ ℝ^(r* × d_in)
    - B = [B₁, B₂, ..., Bₖ]，每个 Bₖ ∈ ℝ^(d_out × r*)
    其中 r* = r / K
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = 4,
        total_rank: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        
        assert total_rank % num_experts == 0, \
            f"total_rank ({total_rank}) must be divisible by num_experts ({num_experts})"
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.total_rank = total_rank
        self.expert_rank = total_rank // num_experts  # r* = r/K
        self.dtype = dtype
        
        # 缩放因子 (使用 rsLoRA 风格的缩放)
        self.scaling = lora_alpha / math.sqrt(total_rank)
        
        # Dropout
        self.dropout = nn.Dropout(lora_dropout)
        
        # 专家参数 - 按 rank 拆分
        # A 矩阵: [num_experts, expert_rank, in_features]
        self.lora_A = nn.Parameter(
            torch.empty((num_experts, self.expert_rank, in_features), dtype=dtype)
        )
        # B 矩阵: [num_experts, out_features, expert_rank]
        self.lora_B = nn.Parameter(
            torch.empty((num_experts, out_features, self.expert_rank), dtype=dtype)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        # Kaiming 初始化 A 矩阵
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
        # B 矩阵初始化为 0
        nn.init.zeros_(self.lora_B)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, in_features]
            routing_weights: [B, T, num_experts] 或 [B, num_experts] 
                            - 稀疏权重（只有 top-k 非零）
            expert_indices: [B, T, top_k] 或 [B, top_k] (可选，用于稀疏计算)
            
        Returns:
            output: [B, T, out_features]
        """
        hidden_states = self.dropout(hidden_states)
        batch_size, seq_len, _ = hidden_states.shape
        
        # 扩展序列级路由权重到 token 级
        if routing_weights.dim() == 2:
            routing_weights = routing_weights.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 方法1: 密集计算（适用于所有专家权重都需要的情况）
        output = self._forward_dense(hidden_states, routing_weights)
        
        return output
    
    def _forward_dense(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        密集计算：计算所有专家然后加权求和
        
        Args:
            hidden_states: [B, T, D_in]
            routing_weights: [B, T, num_experts]
            
        Returns:
            output: [B, T, D_out]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. 通过所有专家的 A 矩阵
        # hidden_states: [B, T, D_in]
        # lora_A: [E, r*, D_in]
        # 结果: [B, T, E, r*]
        h = torch.einsum('btd,erd->bter', hidden_states, self.lora_A)
        
        # 2. 加权（在专家维度）
        # routing_weights: [B, T, E] -> [B, T, E, 1]
        h = h * routing_weights.unsqueeze(-1)  # [B, T, E, r*]
        
        # 3. 通过所有专家的 B 矩阵
        # lora_B: [E, D_out, r*]
        # h: [B, T, E, r*]
        # 结果: [B, T, E, D_out]
        output = torch.einsum('bter,eor->bteo', h, self.lora_B)
        
        # 4. 跨专家求和
        output = output.sum(dim=2)  # [B, T, D_out]
        
        # 5. 应用缩放
        output = output * self.scaling
        
        return output
    
    def _forward_sparse(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        稀疏计算：只计算被选中的专家（更高效但实现更复杂）
        
        Args:
            hidden_states: [B, T, D_in]
            routing_weights: [B, T, top_k]
            expert_indices: [B, T, top_k]
            
        Returns:
            output: [B, T, D_out]
        """
        batch_size, seq_len, top_k = expert_indices.shape
        
        # 展平 batch 和 seq 维度以便处理
        hidden_flat = hidden_states.view(-1, self.in_features)  # [B*T, D_in]
        indices_flat = expert_indices.view(-1, top_k)  # [B*T, top_k]
        weights_flat = routing_weights.view(-1, top_k)  # [B*T, top_k]
        
        output = torch.zeros(
            batch_size * seq_len, self.out_features,
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        
        for k in range(top_k):
            expert_idx = indices_flat[:, k]  # [B*T]
            weight = weights_flat[:, k:k+1]  # [B*T, 1]
            
            # 获取对应专家的参数
            A_k = self.lora_A[expert_idx]  # [B*T, r*, D_in]
            B_k = self.lora_B[expert_idx]  # [B*T, D_out, r*]
            
            # 计算 LoRA 输出
            h = torch.bmm(A_k, hidden_flat.unsqueeze(-1)).squeeze(-1)  # [B*T, r*]
            out = torch.bmm(B_k, h.unsqueeze(-1)).squeeze(-1)  # [B*T, D_out]
            
            output += weight * out
        
        output = output.view(batch_size, seq_len, self.out_features) * self.scaling
        
        return output


class MoELoRALinear(nn.Module):
    """
    完整的 MoE-LoRA Linear 层，包含基础权重和 MoE-LoRA adapter
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        config,  # MoELoRAConfig
        router: Optional['ExpertRouter'] = None,
        share_router: bool = False
    ):
        super().__init__()
        
        # 基础层参数（冻结）
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.weight = base_layer.weight
        self.bias = base_layer.bias if hasattr(base_layer, 'bias') else None
        
        # 确定 dtype
        dtype = getattr(torch, config.torch_dtype) if isinstance(config.torch_dtype, str) else config.torch_dtype
        
        # MoE-LoRA 层
        self.moe_lora = MoELoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            num_experts=config.num_experts,
            total_rank=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            dtype=dtype
        )
        
        # Router (可共享)
        self.share_router = share_router
        if not share_router:
            from .router import ExpertRouter
            self.router = ExpertRouter(
                input_dim=self.in_features,
                num_experts=config.num_experts,
                top_k=config.top_k,
                router_type=config.router_type,
                hidden_dim=config.router_hidden_dim,
                dtype=dtype
            )
        else:
            self.router = router
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        external_routing_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, D]
            external_routing_weights: 外部提供的路由权重（用于共享 router 的情况）
            
        Returns:
            output: [B, T, D_out]
        """
        # 1. 基础层前向
        base_output = F.linear(hidden_states, self.weight, self.bias)
        
        # 2. 获取路由权重
        if external_routing_weights is not None:
            routing_weights = external_routing_weights
            expert_indices = None
        else:
            routing_weights, expert_indices = self.router(
                hidden_states, return_all_weights=True
            )
        
        # 3. MoE-LoRA 前向
        lora_output = self.moe_lora(hidden_states, routing_weights, expert_indices)
        
        # 4. 合并输出
        return base_output + lora_output
    
    def get_routing_loss(self) -> torch.Tensor:
        """获取路由负载均衡损失"""
        if hasattr(self, 'router') and self.router is not None:
            return self.router.load_balancing_loss()
        return torch.tensor(0.0)
    
import re
import types
import json
import os
import torch
import torch.nn as nn
from typing import List, Optional, Dict

def get_peft_model_moe_lora(model: nn.Module, config) -> nn.Module:
    """
    将 MoE-LoRA 应用到模型
    
    Args:
        model: 预训练模型
        config: MoELoRAConfig
        
    Returns:
        应用了 MoE-LoRA 的模型
    """
    # from .moe_lora import MoELoRALinear
    # from .router import ExpertRouter
    
    # 确定 dtype
    dtype = getattr(torch, config.torch_dtype) if isinstance(config.torch_dtype, str) else config.torch_dtype
    
    # 收集所有的 router 和 MoE-LoRA 层
    routers = nn.ModuleList()
    moe_lora_layers = nn.ModuleList()
    
    def _find_and_replace_modules(parent_module: nn.Module, target_modules: List[str]):
        """递归查找并替换目标模块"""
        for name, child in parent_module.named_children():
            # 检查是否是目标模块
            if any(target in name for target in target_modules):
                if isinstance(child, nn.Linear):
                    # 创建 MoE-LoRA 层
                    moe_lora_layer = MoELoRALinear(child, config)
                    setattr(parent_module, name, moe_lora_layer)
                    
                    # 收集组件
                    routers.append(moe_lora_layer.router)
                    moe_lora_layers.append(moe_lora_layer)
                    
                    print(f"Replaced {name} with MoE-LoRA (experts={config.num_experts}, "
                          f"rank_per_expert={config.lora_r // config.num_experts})")
            else:
                # 递归处理子模块
                _find_and_replace_modules(child, target_modules)
    
    # 应用替换
    _find_and_replace_modules(model, config.target_modules)
    
    # 冻结基础模型参数
    for param_name, param in model.named_parameters():
        if not any(key in param_name for key in ['lora_A', 'lora_B', 'router']):
            param.requires_grad = False
    
    # 添加 router manager
    model.router_manager = RouterManager(routers, moe_lora_layers, config)
    
    # 添加 save_pretrained 方法
    model.save_pretrained = types.MethodType(_save_pretrained, model)
    model.peft_config = config
    
    # 打印可训练参数统计
    _print_trainable_parameters(model)
    
    return model


class RouterManager(nn.Module):
    """管理所有 Router 和损失计算"""
    
    def __init__(
        self,
        routers: nn.ModuleList,
        moe_lora_layers: nn.ModuleList,
        config
    ):
        super().__init__()
        self.routers = routers
        self.moe_lora_layers = moe_lora_layers
        self.load_balance_weight = config.load_balance_weight
        self.use_load_balancing_loss = config.use_load_balancing_loss
    
    def get_auxiliary_loss(self) -> torch.Tensor:
        """计算所有 router 的负载均衡损失"""
        if not self.use_load_balancing_loss:
            return torch.tensor(0.0)
        
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.moe_lora_layers:
            total_loss += layer.get_routing_loss()
        
        return total_loss * self.load_balance_weight / len(self.moe_lora_layers)
    
    def clear(self):
        """清除所有 router 的缓存"""
        for router in self.routers:
            router.clear()


def _save_pretrained(self: nn.Module, path: str):
    """保存 MoE-LoRA 权重"""
    if not os.path.exists(path):
        os.makedirs(path)
    
    # 保存可训练参数
    trainable_params = {}
    for name, param in self.named_parameters():
        if param.requires_grad:
            trainable_params[name] = param.detach().cpu()
    
    torch.save(trainable_params, os.path.join(path, 'adapter_model.safetensors'))
    
    # 保存配置
    config_dict = vars(self.peft_config) if hasattr(self.peft_config, '__dict__') else {}
    config_dict['torch_dtype'] = str(config_dict.get('torch_dtype', 'bfloat16'))
    
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Saved MoE-LoRA weights to {path}")


def _print_trainable_parameters(model: nn.Module):
    """打印可训练参数统计"""
    trainable_params = 0
    all_params = 0
    
    for param_name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"Trainable parameters: {trainable_params:,} / {all_params:,} "
          f"({100 * trainable_params / all_params:.2f}%)")