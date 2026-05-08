import argparse
import datetime
import math
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import eigsh
import torch.nn.functional as F
from torch.nn import Parameter
from preprocessing.build_graph import load_graph_adj_mtx, load_graph_node_features

# def parameter_parser():
#     parser = argparse.ArgumentParser(description="Run GETNext.")
#     parser.add_argument('--seed',
#                         type=int,
#                         default=42,
#                         help='Random seed')
#     # parser.add_argument('--device',
#     #                     type=str,
#     #                     default=device,
#     #                     help='')
#     # Data
#     parser.add_argument('--data-adj-mtx',
#                         type=str,
#                         default='../datasets/nyc/preprocessed/graph_A.csv',
#                         help='Graph adjacent path')
#     parser.add_argument('--data-node-feats',
#                         type=str,
#                         default='../datasets/nyc/preprocessed/graph_X.csv',
#                         help='Graph node features path')
#     parser.add_argument('--data-train',
#                         type=str,
#                         default='../datasets/nyc/preprocessed/train_sample.csv',
#                         help='Training data path')
#     parser.add_argument('--time-units',
#                         type=int,
#                         default=48,
#                         help='Time unit is 0.5 hour, 24/0.5=48')
#     parser.add_argument('--time-feature',
#                         type=str,
#                         default='norm_in_day_time',
#                         help='The name of time feature in the data')

#     # Model hyper-parameters
#     parser.add_argument('--poi-embed-dim',
#                         type=int,
#                         default=128,
#                         help='POI embedding dimensions')
#     parser.add_argument('--gcn-dropout',
#                         type=float,
#                         default=0.3,
#                         help='Dropout rate for gcn')
#     parser.add_argument('--gcn-nhid',
#                         type=list,
#                         default=[32, 64],
#                         help='List of hidden dims for gcn layers')
#     parser.add_argument('--time-embed-dim',
#                         type=int,
#                         default=128,
#                         help='Time embedding dimensions')
#     parser.add_argument('--node-attn-nhid',
#                         type=int,
#                         default=128,
#                         help='Node attn map hidden dimensions')
#     return parser.parse_args()

class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        Wh = torch.mm(X, self.W)

        e = self._prepare_attentional_mechanism_input(Wh)

        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # mask

        A = A + 1  # shift from 0-1 to 1-2
        e = e * A

        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout = 0.01):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x

def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]

    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    if mat_type == 'com_lap_mat':
        # Combinatorial
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        # For ChebConv
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        # For GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')
    
    
    
    
def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], dim=2)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


# args = parameter_parser()
#     # The name of node features in NYC/graph_X.csv
# args.feature1 = 'latitude'
# args.feature2 = 'longitude'
# raw_A = load_graph_adj_mtx(args.data_adj_mtx)
# raw_X = load_graph_node_features(args.data_node_feats,
#                                      args.feature1,
#                                      args.feature2,
#                                 )
# num_pois = raw_X.shape[0]
# print(f"num_pois:num_pois")
# X = raw_X
# print('Laplician matrix...')
# A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')

# if isinstance(X, np.ndarray):
#     X = torch.from_numpy(X)
#     A = torch.from_numpy(A)
# X = X.to(device=args.device, dtype=torch.float)
# A = A.to(device=args.device, dtype=torch.float)

# args.gcn_nfeat = X.shape[1]
# poi_embed_model = GCN(ninput=args.gcn_nfeat,
#                         nhid=args.gcn_nhid,
#                         noutput=args.poi_embed_dim,
#                         dropout=args.gcn_dropout)

# time_embed_model = Time2Vec('sin', out_dim=args.time_embed_dim)
# # Node Attn Model
# # node_attn_model = NodeAttnMap(in_features=X.shape[1], nhid=args.node_attn_nhid, use_mask=False)

# poi_embeddings = poi_embed_model(X, A).detach().cpu().numpy()


# def get_history_embedding(historical_trajectories, tokenizer, model):
#     hidden_size = model.config.hidden_size
#     # 时间向量的维度（如 128） → Qwen 的维度（如 2048）
#     time_proj = nn.Linear(args.time_embed_dim, hidden_size).to(device)
#     # POI 向量的维度（如 GCN 输出 128） → Qwen 的维度
#     poi_proj = nn.Linear(args.poi_embed_dim, hidden_size).to(device)
#     history_vectors = []

#     for _, traj_df in historical_trajectories:
#         for _, row in traj_df.iterrows():
#             # ---------- 1. 时间嵌入 ----------
#             dt = datetime.strptime(row['UTCTimeOffset'], "%Y-%m-%d %H:%M:%S")
#             timestamp = torch.tensor([[dt.timestamp()]], dtype=torch.float32).to(device)
#             t_emb = time_embed_model(timestamp).squeeze(0)
#             t_emb_proj = time_proj(t_emb)  # → (hidden_size,)


#             # ---------- 2. POI 嵌入 ----------
#             poi_id = int(row['PoiId'])
#             p_emb = poi_embeddings[poi_id]  # 由训练好的 GCN 生成的全局向量
#             p_emb_proj = poi_proj(p_emb)  # → (hidden_size,)

#             # ---------- 3. 类别嵌入 ----------
#             category_name = row['PoiCategoryName']
#             tokens = tokenizer(category_name, return_tensors="pt", add_special_tokens=False).to(device)
#             input_ids = tokens["input_ids"]  # shape: (1, seq_len)
#             with torch.no_grad():
#                 embedded_tokens = model.model.embed_tokens(input_ids)  # shape: (1, seq_len, hidden_size)
#             # 3. 平均池化（也可以选第一个 token）
#             c_emb = embedded_tokens.mean(dim=1).squeeze(0)  # shape: (hidden_size,)

#             # ---------- 4. 拼接 ----------
#             emb = torch.cat([t_emb_proj, p_emb_proj, c_emb], dim=-1)  # → (hidden_size * 3,)
#             history_vectors.append(emb)

#     if history_vectors:
#         return torch.stack(history_vectors, dim=0)  # (seq_len, total_dim)
#     else:
#         return torch.zeros((0, hidden_size * 3), dtype=torch.bfloat16).to(device)