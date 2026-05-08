# stkg_retriever/train_retriever.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
from pathlib import Path

from data_processor import POIDataProcessor
from stkg_builder import STKGBuilder
from gnn_encoder import STKGEncoder, SubGraphSampler
from retriever import STKGRetriever, STKGContrastiveLoss


class TrajectoryDataset(Dataset):
    """轨迹数据集"""
    
    def __init__(self, trajectories, stkg, max_len=50):
        self.trajectories = trajectories
        self.stkg = stkg
        self.max_len = max_len
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        
        user_id = traj['user_id']
        checkins = traj['checkins'][-self.max_len:]
        
        # 获取实体ID
        user_entity = self.stkg.entity2id[f"user_{user_id}"]
        poi_entities = [
            self.stkg.entity2id[f"poi_{c['poi_id']}"]
            for c in checkins
        ]
        
        # 目标：预测最后一个POI
        target_poi = poi_entities[-1]
        context_entities = [user_entity] + poi_entities[:-1]
        
        return {
            'user_entity': user_entity,
            'context_entities': context_entities,
            'target_poi': target_poi,
            'trajectory_id': traj['trajectory_id']
        }


def train_epoch(
    model,
    stkg_encoder,
    dataloader,
    optimizer,
    edge_index,
    edge_type,
    sampler,
    contrastive_loss_fn,
    device
):
    model.train()
    stkg_encoder.train()
    
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        # 为每个轨迹采样两个子图视图
        batch_view1_embeds = []
        batch_view2_embeds = []
        batch_targets = []
        
        for i in range(len(batch['user_entity'])):
            context = batch['context_entities'][i]
            target = batch['target_poi'][i]
            
            # 采样两个视图
            nodes1, ei1, et1 = sampler.sample(context)
            nodes2, ei2, et2 = sampler.sample(context)
            
            # 编码
            embed1 = stkg_encoder(ei1.to(device), et1.to(device), nodes1.to(device))
            embed2 = stkg_encoder(ei2.to(device), et2.to(device), nodes2.to(device))
            
            # Mean pooling
            batch_view1_embeds.append(embed1.mean(dim=0))
            batch_view2_embeds.append(embed2.mean(dim=0))
            batch_targets.append(target)
        
        view1_embeds = torch.stack(batch_view1_embeds)
        view2_embeds = torch.stack(batch_view2_embeds)
        targets = torch.LongTensor(batch_targets).to(device)
        
        # 对比学习损失
        cl_loss = contrastive_loss_fn(view1_embeds, view2_embeds)
        
        # 检索损失（使用融合的表示作为查询）
        query_embeds = (view1_embeds + view2_embeds) / 2
        retrieval_loss = model(query_embeds, targets, edge_index, edge_type)
        
        # 总损失
        loss = cl_loss + retrieval_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/train_sample.csv')
    parser.add_argument('--save_dir', type=str, default='checkpoints/stkg')
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. 数据处理
    print("Processing data...")
    processor = POIDataProcessor()
    data = processor.load_and_process(args.data_path)
    processor.save(f"{args.save_dir}/processor.pkl")
    
    # 2. 构建STKG
    print("Building STKG...")
    stkg = STKGBuilder()
    stkg.build(data)
    stkg.save(f"{args.save_dir}/stkg.pkl")
    
    # 3. 准备图数据
    triplets = stkg.get_triplets_tensor()
    edge_index = triplets[:, [0, 2]].t().to(device)
    edge_type = triplets[:, 1].to(device)
    
    # 4. 初始化模型
    stkg_encoder = STKGEncoder(
        num_entities=len(stkg.entity2id),
        num_relations=len(stkg.relation2id),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
    retriever = STKGRetriever(
        stkg_encoder=stkg_encoder,
        stkg=stkg,
        embed_dim=args.embed_dim
    ).to(device)
    
    sampler = SubGraphSampler(stkg, num_hops=2, num_neighbors=10)
    contrastive_loss = STKGContrastiveLoss(temperature=0.5)
    
    # 5. 准备数据加载器
    dataset = TrajectoryDataset(data['trajectories'], stkg)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 6. 优化器
    optimizer = torch.optim.Adam(
        list(stkg_encoder.parameters()) + list(retriever.parameters()),
        lr=args.lr
    )
    
    # 7. 训练
    print("Training...")
    for epoch in range(args.epochs):
        loss = train_epoch(
            retriever, stkg_encoder, dataloader, optimizer,
            edge_index, edge_type, sampler, contrastive_loss, device
        )
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'encoder': stkg_encoder.state_dict(),
                'retriever': retriever.state_dict()
            }, f"{args.save_dir}/model_epoch{epoch+1}.pt")
    
    # 8. 构建检索索引
    retriever.build_poi_index(edge_index, edge_type)
    torch.save({
        'encoder': stkg_encoder.state_dict(),
        'retriever': retriever.state_dict(),
        'poi_embeddings': retriever.poi_embeddings,
        'poi_entity_ids': retriever.poi_entity_ids
    }, f"{args.save_dir}/final_model.pt")
    
    print("Training completed!")


if __name__ == '__main__':
    main()