# stkg_retriever/stkg_builder.py
import torch
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict
from typing import List, Dict, Tuple
import pickle

class STKGBuilder:
    """空间-时间知识图谱构建器"""
    
    # 关系类型定义
    RELATION_TYPES = {
        'user_checkin': 0,       # 用户签到POI
        'poi_category': 1,       # POI属于某类别
        'poi_region': 2,         # POI位于某区域
        'category_similar': 3,   # 类别相似
        'spatial_near': 4,       # 空间邻近
        # 时空转移关系会动态添加
    }
    
    def __init__(self, time_unit=0.5, distance_unit=100, max_time_interval=48, max_dist_interval=50):
        """
        Args:
            time_unit: 时间单位（小时）
            distance_unit: 距离单位（米）
            max_time_interval: 最大时间间隔单位数
            max_dist_interval: 最大距离间隔单位数
        """
        self.time_unit = time_unit
        self.distance_unit = distance_unit
        self.max_time_interval = max_time_interval
        self.max_dist_interval = max_dist_interval
        
        # 实体映射（统一编号）
        self.entity2id = {}
        self.id2entity = {}
        self.entity_type = {}  # entity_id -> type ('user', 'poi', 'category', 'region')
        
        # 关系映射
        self.relation2id = dict(self.RELATION_TYPES)
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.next_relation_id = len(self.RELATION_TYPES)
        
        # 三元组存储
        self.triplets = []  # [(head, relation, tail), ...]
        
        # 邻接表
        self.head_to_tail = defaultdict(list)  # head -> [(rel, tail), ...]
        self.tail_to_head = defaultdict(list)  # tail -> [(rel, head), ...]
        
        # 实体数量统计
        self.num_users = 0
        self.num_pois = 0
        self.num_categories = 0
        self.num_regions = 0
        
    def build(self, processed_data: Dict) -> 'STKGBuilder':
        """
        从处理后的数据构建STKG
        
        Args:
            processed_data: POIDataProcessor处理后的数据
        """
        trajectories = processed_data['trajectories']
        poi_info = processed_data['poi_info']
        user2id = processed_data['user2id']
        category2id = processed_data['category2id']
        region2id = processed_data['region2id']
        
        print("Building STKG...")
        
        # 1. 添加所有实体
        self._add_entities(user2id, poi_info, category2id, region2id)
        
        # 2. 添加POI属性关系
        self._add_poi_property_relations(poi_info)
        
        # 3. 从轨迹构建签到关系和时空转移关系
        self._add_trajectory_relations(trajectories, poi_info)
        
        # 4. 添加空间邻近关系
        self._add_spatial_relations(poi_info)
        
        print(f"STKG built: {len(self.entity2id)} entities, "
              f"{len(self.relation2id)} relation types, {len(self.triplets)} triplets")
        
        return self
    
    def _add_entities(self, user2id, poi_info, category2id, region2id):
        """添加所有实体"""
        entity_idx = 0
        
        # 添加用户实体
        for user_id, _ in user2id.items():
            entity_key = f"user_{user_id}"
            self.entity2id[entity_key] = entity_idx
            self.id2entity[entity_idx] = entity_key
            self.entity_type[entity_idx] = 'user'
            entity_idx += 1
        self.num_users = len(user2id)
        
        # 添加POI实体
        for poi_idx, info in poi_info.items():
            entity_key = f"poi_{poi_idx}"
            self.entity2id[entity_key] = entity_idx
            self.id2entity[entity_idx] = entity_key
            self.entity_type[entity_idx] = 'poi'
            entity_idx += 1
        self.num_pois = len(poi_info)
        
        # 添加类别实体
        for cat_id, _ in category2id.items():
            entity_key = f"category_{cat_id}"
            self.entity2id[entity_key] = entity_idx
            self.id2entity[entity_idx] = entity_key
            self.entity_type[entity_idx] = 'category'
            entity_idx += 1
        self.num_categories = len(category2id)
        
        # 添加区域实体
        for region_key, _ in region2id.items():
            entity_key = f"region_{region_key}"
            self.entity2id[entity_key] = entity_idx
            self.id2entity[entity_idx] = entity_key
            self.entity_type[entity_idx] = 'region'
            entity_idx += 1
        self.num_regions = len(region2id)
    
    def _add_poi_property_relations(self, poi_info: Dict):
        """添加POI属性关系"""
        for poi_idx, info in poi_info.items():
            poi_entity = self.entity2id[f"poi_{poi_idx}"]
            
            # POI -> Category
            cat_entity = self.entity2id[f"category_{info['category_id']}"]
            self._add_triplet(poi_entity, self.relation2id['poi_category'], cat_entity)
            
            # POI -> Region
            if 'region_id' in info:
                # 需要找到region的entity_key
                for key, idx in self.entity2id.items():
                    if key.startswith('region_') and self.entity_type[idx] == 'region':
                        # 这里简化处理，实际需要正确映射
                        pass
    
    def _add_trajectory_relations(self, trajectories: List[Dict], poi_info: Dict):
        """从轨迹添加签到关系和时空转移关系"""
        transfer_relation_cache = {}  # (time_interval, dist_interval) -> relation_id
        
        for traj in trajectories:
            user_entity = self.entity2id[f"user_{traj['user_id']}"]
            checkins = traj['checkins']
            
            for i, checkin in enumerate(checkins):
                poi_entity = self.entity2id[f"poi_{checkin['poi_id']}"]
                
                # 用户签到关系
                self._add_triplet(user_entity, self.relation2id['user_checkin'], poi_entity)
                
                # 时空转移关系
                if i > 0:
                    prev_checkin = checkins[i - 1]
                    prev_poi_entity = self.entity2id[f"poi_{prev_checkin['poi_id']}"]
                    
                    # 计算时间间隔
                    time_diff = (checkin['epoch'] - prev_checkin['epoch']) / 3600  # 小时
                    time_interval = min(int(time_diff / self.time_unit), self.max_time_interval)
                    
                    # 计算距离间隔
                    distance = self._haversine(
                        prev_checkin['lat'], prev_checkin['lon'],
                        checkin['lat'], checkin['lon']
                    )
                    dist_interval = min(int(distance / self.distance_unit), self.max_dist_interval)
                    
                    # 获取或创建时空转移关系
                    transfer_key = (time_interval, dist_interval)
                    if transfer_key not in transfer_relation_cache:
                        rel_name = f"transfer_t{time_interval}_d{dist_interval}"
                        rel_id = self.next_relation_id
                        self.relation2id[rel_name] = rel_id
                        self.id2relation[rel_id] = rel_name
                        transfer_relation_cache[transfer_key] = rel_id
                        self.next_relation_id += 1
                    
                    rel_id = transfer_relation_cache[transfer_key]
                    self._add_triplet(prev_poi_entity, rel_id, poi_entity)
    
    def _add_spatial_relations(self, poi_info: Dict, threshold=500):
        """添加空间邻近关系（距离小于阈值的POI对）"""
        poi_list = list(poi_info.items())
        
        for i, (poi_idx1, info1) in enumerate(poi_list):
            for j, (poi_idx2, info2) in enumerate(poi_list[i+1:], i+1):
                distance = self._haversine(
                    info1['lat'], info1['lon'],
                    info2['lat'], info2['lon']
                )
                
                if distance < threshold:
                    poi_entity1 = self.entity2id[f"poi_{poi_idx1}"]
                    poi_entity2 = self.entity2id[f"poi_{poi_idx2}"]
                    
                    # 双向关系
                    self._add_triplet(poi_entity1, self.relation2id['spatial_near'], poi_entity2)
                    self._add_triplet(poi_entity2, self.relation2id['spatial_near'], poi_entity1)
    
    def _add_triplet(self, head: int, relation: int, tail: int):
        """添加三元组"""
        self.triplets.append((head, relation, tail))
        self.head_to_tail[head].append((relation, tail))
        self.tail_to_head[tail].append((relation, head))
    
    def _haversine(self, lat1, lon1, lat2, lon2):
        """计算两点间距离（米）"""
        R = 6371000
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1-a))
    
    def get_triplets_tensor(self) -> torch.LongTensor:
        """获取三元组张量"""
        return torch.LongTensor(self.triplets)
    
    def get_poi_entity_ids(self) -> List[int]:
        """获取所有POI实体ID"""
        return [idx for idx, etype in self.entity_type.items() if etype == 'poi']
    
    def save(self, save_path: str):
        """保存STKG"""
        data = {
            'entity2id': self.entity2id,
            'id2entity': self.id2entity,
            'entity_type': self.entity_type,
            'relation2id': self.relation2id,
            'id2relation': self.id2relation,
            'triplets': self.triplets,
            'head_to_tail': dict(self.head_to_tail),
            'tail_to_head': dict(self.tail_to_head),
            'num_users': self.num_users,
            'num_pois': self.num_pois,
            'num_categories': self.num_categories,
            'num_regions': self.num_regions
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, load_path: str) -> 'STKGBuilder':
        """加载STKG"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        builder = cls()
        for key, value in data.items():
            setattr(builder, key, value)
        
        # 恢复defaultdict
        builder.head_to_tail = defaultdict(list, builder.head_to_tail)
        builder.tail_to_head = defaultdict(list, builder.tail_to_head)
        
        return builder
    
# stkg_retriever/gnn_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, RGATConv
from torch_geometric.data import Data
from typing import Optional, Tuple

class RelationalGATLayer(nn.Module):
    """关系感知的图注意力层"""
    
    def __init__(self, in_dim, out_dim, num_relations, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        # 关系特定的变换
        self.relation_transform = nn.Embedding(num_relations, in_dim * out_dim)
        
        # 注意力参数
        self.W_query = nn.Linear(in_dim, out_dim)
        self.W_key = nn.Linear(in_dim, out_dim)
        self.W_value = nn.Linear(in_dim, out_dim)
        self.W_relation = nn.Linear(in_dim, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
        
    def forward(self, x, edge_index, edge_type, relation_embed):
        """
        Args:
            x: 节点特征 [num_nodes, in_dim]
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
            relation_embed: 关系嵌入 [num_relations, in_dim]
        """
        num_nodes = x.size(0)
        
        # Query, Key, Value变换
        Q = self.W_query(x)  # [num_nodes, out_dim]
        K = self.W_key(x)
        V = self.W_value(x)
        
        # 计算注意力
        src, dst = edge_index[0], edge_index[1]
        
        # 关系嵌入
        rel_embed = relation_embed[edge_type]  # [num_edges, in_dim]
        rel_transform = self.W_relation(rel_embed)  # [num_edges, out_dim]
        
        # 注意力分数: (Q_dst * (K_src + R)) / sqrt(d)
        Q_dst = Q[dst]  # [num_edges, out_dim]
        K_src = K[src]  # [num_edges, out_dim]
        
        attn_scores = (Q_dst * (K_src + rel_transform)).sum(dim=-1) / (self.head_dim ** 0.5)
        
        # Softmax（按目标节点分组）
        attn_weights = self._scatter_softmax(attn_scores, dst, num_nodes)
        attn_weights = self.dropout(attn_weights)
        
        # 聚合
        V_src = V[src]  # [num_edges, out_dim]
        weighted_values = attn_weights.unsqueeze(-1) * (V_src + rel_transform)
        
        # 按目标节点聚合
        out = torch.zeros(num_nodes, V.size(-1), device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_values), weighted_values)
        
        # 残差连接和层归一化
        if x.size(-1) == out.size(-1):
            out = self.layer_norm(out + x)
        else:
            out = self.layer_norm(out)
        
        return out
    
    def _scatter_softmax(self, src, index, num_nodes):
        """分组softmax"""
        max_val = torch.zeros(num_nodes, device=src.device)
        max_val.scatter_reduce_(0, index, src, reduce='amax', include_self=False)
        
        src_exp = torch.exp(src - max_val[index])
        
        sum_exp = torch.zeros(num_nodes, device=src.device)
        sum_exp.scatter_add_(0, index, src_exp)
        
        return src_exp / (sum_exp[index] + 1e-10)


class STKGEncoder(nn.Module):
    """STKG图神经网络编码器"""
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # 实体和关系嵌入
        self.entity_embedding = nn.Embedding(num_entities, embed_dim)
        self.relation_embedding = nn.Embedding(num_relations, embed_dim)
        
        # GNN层
        self.gnn_layers = nn.ModuleList([
            RelationalGATLayer(
                in_dim=embed_dim if i == 0 else hidden_dim,
                out_dim=hidden_dim,
                num_relations=num_relations,
                num_heads=num_heads,
                dropout=dropout
            )
            for i in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
    
    def forward(
        self,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        node_ids: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        Args:
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
            node_ids: 需要编码的节点ID（None表示所有节点）
        
        Returns:
            node_embeddings: 节点嵌入 [num_nodes, embed_dim]
        """
        # 获取节点嵌入
        if node_ids is None:
            x = self.entity_embedding.weight
        else:
            x = self.entity_embedding(node_ids)
        
        # 关系嵌入
        rel_embed = self.relation_embedding.weight
        
        # 多层GNN
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_type, rel_embed)
            x = F.relu(x)
        
        # 输出投影
        x = self.output_proj(x)
        
        return x
    
    def get_poi_embeddings(
        self,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        poi_entity_ids: torch.LongTensor
    ) -> torch.Tensor:
        """获取POI节点的嵌入"""
        all_embeddings = self.forward(edge_index, edge_type)
        return all_embeddings[poi_entity_ids]


class SubGraphSampler:
    """子图采样器（用于对比学习）"""
    
    def __init__(self, stkg, num_hops=2, num_neighbors=10):
        self.stkg = stkg
        self.num_hops = num_hops
        self.num_neighbors = num_neighbors
    
    def sample(self, center_nodes: list) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
        采样以center_nodes为中心的子图
        
        Returns:
            node_ids: 子图节点ID
            edge_index: 子图边索引
            edge_type: 子图边类型
        """
        sampled_nodes = set(center_nodes)
        sampled_edges = []
        
        current_frontier = list(center_nodes)
        
        for _ in range(self.num_hops):
            next_frontier = []
            
            for node in current_frontier:
                neighbors = self.stkg.head_to_tail.get(node, [])
                
                # 随机采样邻居
                if len(neighbors) > self.num_neighbors:
                    indices = torch.randperm(len(neighbors))[:self.num_neighbors].tolist()
                    neighbors = [neighbors[i] for i in indices]
                
                for rel, tail in neighbors:
                    sampled_edges.append((node, rel, tail))
                    if tail not in sampled_nodes:
                        sampled_nodes.add(tail)
                        next_frontier.append(tail)
            
            current_frontier = next_frontier
        
        # 构建返回数据
        node_list = list(sampled_nodes)
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        
        edge_index = []
        edge_type = []
        
        for head, rel, tail in sampled_edges:
            if head in node_to_idx and tail in node_to_idx:
                edge_index.append([node_to_idx[head], node_to_idx[tail]])
                edge_type.append(rel)
        
        if len(edge_index) == 0:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)
        else:
            edge_index = torch.LongTensor(edge_index).t()
            edge_type = torch.LongTensor(edge_type)
        
        return torch.LongTensor(node_list), edge_index, edge_type
    
# stkg_retriever/retriever.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class STKGRetriever(nn.Module):
    """基于STKG的POI检索器"""
    
    def __init__(
        self,
        stkg_encoder: nn.Module,
        stkg,
        embed_dim: int = 256,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.stkg_encoder = stkg_encoder
        self.stkg = stkg
        self.embed_dim = embed_dim
        self.temperature = temperature
        
        # 查询投影（将LLM hidden state映射到STKG空间）
        self.query_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # POI实体ID缓存
        self.poi_entity_ids = None
        self.poi_embeddings = None
        
    def build_poi_index(self, edge_index, edge_type):
        """预计算所有POI的嵌入（用于快速检索）"""
        self.stkg_encoder.eval()
        
        with torch.no_grad():
            all_embeddings = self.stkg_encoder(edge_index, edge_type)
            
            # 获取POI实体ID
            poi_ids = self.stkg.get_poi_entity_ids()
            self.poi_entity_ids = torch.LongTensor(poi_ids)
            self.poi_embeddings = all_embeddings[self.poi_entity_ids]
        
        print(f"Built POI index with {len(poi_ids)} POIs")
    
    def retrieve(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 50,
        exclude_pois: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        检索Top-K候选POI
        
        Args:
            query_embedding: 查询向量 [batch_size, embed_dim] 或 [embed_dim]
            top_k: 返回的候选数量
            exclude_pois: 需要排除的POI ID列表
        
        Returns:
            poi_ids: 候选POI ID [batch_size, top_k]
            scores: 相似度分数 [batch_size, top_k]
        """
        if self.poi_embeddings is None:
            raise RuntimeError("Please call build_poi_index first!")
        
        # 投影查询向量
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        
        query = self.query_proj(query_embedding)  # [batch_size, embed_dim]
        query = F.normalize(query, dim=-1)
        
        # 计算相似度
        poi_embeds = F.normalize(self.poi_embeddings, dim=-1)
        scores = torch.matmul(query, poi_embeds.t())  # [batch_size, num_pois]
        
        # 排除已访问的POI
        if exclude_pois is not None:
            for poi_id in exclude_pois:
                if poi_id < scores.size(1):
                    scores[:, poi_id] = float('-inf')
        
        # Top-K
        top_scores, top_indices = torch.topk(scores, k=min(top_k, scores.size(1)), dim=-1)
        top_poi_ids = self.poi_entity_ids[top_indices]
        
        return top_poi_ids, top_scores
    
    def forward(
        self,
        query_embedding: torch.Tensor,
        positive_pois: torch.LongTensor,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor
    ) -> torch.Tensor:
        """
        训练时的前向传播（对比学习）
        
        Args:
            query_embedding: 查询向量（来自LLM）[batch_size, embed_dim]
            positive_pois: 正样本POI ID [batch_size]
            edge_index, edge_type: 图结构
        
        Returns:
            loss: 对比学习损失
        """
        # 编码所有节点
        all_embeddings = self.stkg_encoder(edge_index, edge_type)
        
        # 获取POI嵌入
        poi_ids = self.stkg.get_poi_entity_ids()
        poi_embeddings = all_embeddings[poi_ids]  # [num_pois, embed_dim]
        
        # 投影查询
        query = self.query_proj(query_embedding)  # [batch_size, embed_dim]
        query = F.normalize(query, dim=-1)
        poi_embeddings = F.normalize(poi_embeddings, dim=-1)
        
        # 计算相似度
        logits = torch.matmul(query, poi_embeddings.t()) / self.temperature
        
        # 构建标签（正样本POI的索引）
        poi_id_to_idx = {pid: idx for idx, pid in enumerate(poi_ids)}
        labels = torch.LongTensor([poi_id_to_idx[p.item()] for p in positive_pois])
        labels = labels.to(query.device)
        
        # 交叉熵损失
        loss = F.cross_entropy(logits, labels)
        
        return loss


class STKGContrastiveLoss(nn.Module):
    """STKG子图对比学习损失"""
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        view1_embeddings: torch.Tensor,
        view2_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            view1_embeddings: 视图1的轨迹表示 [batch_size, embed_dim]
            view2_embeddings: 视图2的轨迹表示 [batch_size, embed_dim]
        """
        batch_size = view1_embeddings.size(0)
        
        # 归一化
        view1 = F.normalize(view1_embeddings, dim=-1)
        view2 = F.normalize(view2_embeddings, dim=-1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(view1, view2.t()) / self.temperature
        
        # 正样本在对角线上
        labels = torch.arange(batch_size, device=view1.device)
        
        # 双向对比损失
        loss_12 = F.cross_entropy(sim_matrix, labels)
        loss_21 = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_12 + loss_21) / 2