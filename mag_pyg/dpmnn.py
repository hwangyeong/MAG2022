import argparse
import glob
from multiprocessing.sharedctypes import Value
import os
import os.path as osp
from re import S
import time
import copy
from typing import List, NamedTuple, Optional, Tuple
from xmlrpc.client import Boolean

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy
from torch import BoolTensor, Tensor
from torch.nn import BatchNorm1d, LayerNorm, Dropout, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import StepLR
from torch_geometric.utils import subgraph, dropout_adj
from torch_geometric.typing import OptTensor

from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import GATConv, SAGEConv, MessagePassing, GCNConv, GINEConv
from torch_sparse import SparseTensor
from torch_geometric.typing import Adj, OptPairTensor
from tqdm import tqdm

from root import ROOT

class OneGraphNeighborSampler(NeighborSampler):
    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        n_id = batch
        sub_nodes = [n_id]
        for size in self.sizes: #逐层找邻居
            neis = self.adj_t.sample(size, n_id) # (num_nid, 25)
            n_id = neis.flatten()
            sub_nodes.append(n_id)
        sub_nodes = torch.cat(sub_nodes, dim=0).unique()
        row, col, value = self.adj_t.coo()
        edge_index = torch.stack([row, col], dim=0)

        subg_edge_index, weights = subgraph(sub_nodes, edge_index, value, relabel_nodes=True)
        subg = SparseTensor(row=subg_edge_index[0], col=subg_edge_index[1], value=weights,
                       sparse_sizes=(sub_nodes.shape[0], sub_nodes.shape[0]),
                       is_sorted=True)

        center_mask = torch.from_numpy(np.in1d(sub_nodes.numpy(), batch.numpy()))

        # out = (batch_size, n_id, subg, center_mask)
        out = (batch, sub_nodes, subg, weights, center_mask)
        out = self.transform(*out) if self.transform is not None else out
        return out

class Batch(NamedTuple):
    x: Tensor
    edge_x: Tensor
    y: Tensor
    # adjs_t: List[SparseTensor]
    adj_t: SparseTensor
    center_mask: BoolTensor

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            edge_x=self.edge_x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            adj_t=self.adj_t.to(*args, **kwargs),
            center_mask=self.center_mask.to(*args, **kwargs)
        )

    def new(self):
        return Batch(
            x=torch.tensor([]),
            edge_x=torch.tensor([]),
            y=torch.tensor([]),
            adj_t=torch.tensor([]),
            center_mask=torch.tensor([])
        )


class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new

def filter_adj(row: Tensor, col: Tensor, weight: OptTensor, edge_attr: OptTensor,
               mask: Tensor) -> Tuple[Tensor, Tensor, OptTensor, OptTensor]:
    return row[mask], col[mask], \
            None if weight is None else weight[mask], \
            None if edge_attr is None else edge_attr[mask]

def dropout_adj(adj, edge_attr, p=0.5, training=True):

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    row, col, weight = adj.coo()

    mask = torch.rand(row.size(0), device=adj.device()) >= p

    # if force_undirected:
    #     mask[row > col] = False

    row, col, weight, edge_attr = filter_adj(row, col, weight, edge_attr, mask)

    adj_drop = SparseTensor(row=row, col=col, value=weight,
                       sparse_sizes=adj.sparse_sizes(),
                       is_sorted=True)

    return adj_drop, edge_attr


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def get_col_slice(x, start_row_idx, end_row_idx, start_col_idx, end_col_idx):
    outs = []
    chunk = 100000
    for i in tqdm(range(start_row_idx, end_row_idx, chunk)):
        j = min(i + chunk, end_row_idx)
        outs.append(x[i:j, start_col_idx:end_col_idx].copy())
    return np.concatenate(outs, axis=0)


def save_col_slice(x_src, x_dst, start_row_idx, end_row_idx, start_col_idx,
                   end_col_idx):
    assert x_src.shape[0] == end_row_idx - start_row_idx
    assert x_src.shape[1] == end_col_idx - start_col_idx
    chunk, offset = 100000, start_row_idx
    for i in tqdm(range(0, end_row_idx - start_row_idx, chunk)):
        j = min(i + chunk, end_row_idx - start_row_idx)
        x_dst[offset + i:offset + j, start_col_idx:end_col_idx] = x_src[i:j]


class MAG240M(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, sizes: List[int],
                 in_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sizes = sizes
        self.in_memory = in_memory
        # self.aug_params = [0.4, 0.2, 0.4, 0.2]

    @property
    def num_features(self) -> int:
        return 768

    @property
    def num_classes(self) -> int:
        # return 153
        return 150

    @property
    def num_relations(self) -> int:
        return 5

    def prepare_data(self):
        dataset = MAG240MDataset(self.data_dir)

        path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        if not osp.exists(path):  # Will take approximately 5 minutes...
            t = time.perf_counter()
            print('Converting adjacency matrix...', end=' ', flush=True)
            edge_index = dataset.edge_index('paper', 'cites', 'paper')
            edge_index = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                sparse_sizes=(dataset.num_papers, dataset.num_papers),
                is_sorted=True)
            torch.save(adj_t.to_symmetric(), path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        path = f'{dataset.dir}/full_adj_t.pt'
        if not osp.exists(path):  # Will take approximately 16 minutes...
            t = time.perf_counter()
            print('Merging adjacency matrices...', end=' ', flush=True)

            row, col, _ = torch.load(
                f'{dataset.dir}/paper_to_paper_symmetric.pt').coo()
            rows, cols = [row], [col]

            edge_index = dataset.edge_index('author', 'writes', 'paper')
            row, col = torch.from_numpy(edge_index)
            row += dataset.num_papers
            rows += [row, col]
            cols += [col, row]

            edge_index = dataset.edge_index('author', 'institution')
            row, col = torch.from_numpy(edge_index)
            row += dataset.num_papers
            col += dataset.num_papers + dataset.num_authors
            rows += [row, col]
            cols += [col, row]

            edge_types = [
                torch.full(x.size(), i, dtype=torch.int8)
                for i, x in enumerate(rows)
            ]

            row = torch.cat(rows, dim=0)
            del rows
            col = torch.cat(cols, dim=0)
            del cols

            N = (dataset.num_papers + dataset.num_authors +
                 dataset.num_institutions)

            perm = (N * row).add_(col).numpy().argsort()
            perm = torch.from_numpy(perm)
            row = row[perm]
            col = col[perm]

            edge_type = torch.cat(edge_types, dim=0)[perm]
            del edge_types

            full_adj_t = SparseTensor(row=row, col=col, value=edge_type,
                                      sparse_sizes=(N, N), is_sorted=True)

            torch.save(full_adj_t, path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        path = f'{dataset.dir}/full_feat.npy'
        done_flag_path = f'{dataset.dir}/full_feat_done.txt'
        if not osp.exists(done_flag_path):  # Will take ~3 hours...
            t = time.perf_counter()
            print('Generating full feature matrix...')

            node_chunk_size = 100000
            dim_chunk_size = 64
            N = (dataset.num_papers + dataset.num_authors +
                 dataset.num_institutions)

            paper_feat = dataset.paper_feat
            x = np.memmap(path, dtype=np.float16, mode='w+',
                          shape=(N, self.num_features))

            print('Copying paper features...')
            for i in tqdm(range(0, dataset.num_papers, node_chunk_size)):
                j = min(i + node_chunk_size, dataset.num_papers)
                x[i:j] = paper_feat[i:j]

            edge_index = dataset.edge_index('author', 'writes', 'paper')
            row, col = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=row, col=col,
                sparse_sizes=(dataset.num_authors, dataset.num_papers),
                is_sorted=True)

            # Processing 64-dim subfeatures at a time for memory efficiency.
            print('Generating author features...')
            for i in tqdm(range(0, self.num_features, dim_chunk_size)):
                j = min(i + dim_chunk_size, self.num_features)
                inputs = get_col_slice(paper_feat, start_row_idx=0,
                                       end_row_idx=dataset.num_papers,
                                       start_col_idx=i, end_col_idx=j)
                inputs = torch.from_numpy(inputs)
                outputs = adj_t.matmul(inputs, reduce='mean').numpy()
                del inputs
                save_col_slice(
                    x_src=outputs, x_dst=x, start_row_idx=dataset.num_papers,
                    end_row_idx=dataset.num_papers + dataset.num_authors,
                    start_col_idx=i, end_col_idx=j)
                del outputs

            edge_index = dataset.edge_index('author', 'institution')
            row, col = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=col, col=row,
                sparse_sizes=(dataset.num_institutions, dataset.num_authors),
                is_sorted=False)

            print('Generating institution features...')
            # Processing 64-dim subfeatures at a time for memory efficiency.
            for i in tqdm(range(0, self.num_features, dim_chunk_size)):
                j = min(i + dim_chunk_size, self.num_features)
                inputs = get_col_slice(
                    x, start_row_idx=dataset.num_papers,
                    end_row_idx=dataset.num_papers + dataset.num_authors,
                    start_col_idx=i, end_col_idx=j)
                inputs = torch.from_numpy(inputs)
                outputs = adj_t.matmul(inputs, reduce='mean').numpy()
                del inputs
                save_col_slice(
                    x_src=outputs, x_dst=x,
                    start_row_idx=dataset.num_papers + dataset.num_authors,
                    end_row_idx=N, start_col_idx=i, end_col_idx=j)
                del outputs

            x.flush()
            del x
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

            with open(done_flag_path, 'w') as f:
                f.write('done')

    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()
        print('Reading dataset...', end=' ', flush=True)
        dataset = MAG240MDataset(self.data_dir)

        self.train_idx = torch.from_numpy(dataset.get_idx_split('train'))
        self.train_idx = self.train_idx
        self.train_idx.share_memory_()
        self.val_idx = torch.from_numpy(dataset.get_idx_split('valid'))
        self.val_idx.share_memory_()
        # self.test_idx = torch.from_numpy(dataset.get_idx_split('test-dev'))
        self.test_idx = torch.from_numpy(dataset.get_idx_split('test'))
        self.test_idx.share_memory_()

        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions

        x = np.memmap(f'{dataset.dir}/full_feat.npy', dtype=np.float16,
                      mode='r', shape=(N, self.num_features))

        if self.in_memory:
            self.x = np.empty((N, self.num_features), dtype=np.float16)
            self.x[:] = x
            self.x = torch.from_numpy(self.x).share_memory_()
        else:
            self.x = x

        self.y = torch.from_numpy(dataset.all_paper_label)

        path = f'{dataset.dir}/full_adj_t.pt'
        self.adj_t = torch.load(path)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    def train_dataloader(self):
        # return NeighborSampler(self.adj_t, node_idx=self.train_idx,
        #                        sizes=self.sizes, return_e_id=False,
        #                        transform=self.convert_batch,
        #                        batch_size=self.batch_size, shuffle=True,
        #                        num_workers=32)
        return OneGraphNeighborSampler(self.adj_t, node_idx=self.train_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, shuffle=True,
                               num_workers=16)

    def val_dataloader(self):
        # return NeighborSampler(self.adj_t, node_idx=self.val_idx,
        #                        sizes=self.sizes, return_e_id=False,
        #                        transform=self.convert_batch,
        #                        batch_size=self.batch_size, num_workers=1)
        return OneGraphNeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=16)

    def test_dataloader(self):  # Test best validation model once again.
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=16)

    def hidden_test_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.test_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=24)

    def convert_batch(self, batch, n_id, adj, weights, center_mask):
        if self.in_memory:
            x = self.x[n_id].to(torch.float)
        else:
            x = torch.from_numpy(self.x[n_id.numpy()]).to(torch.float)

        y = self.y[batch].to(torch.long)

        weights = weights.long()
        edge_feats = torch.zeros(weights.shape[0], 7).scatter_(1, weights.unsqueeze(1), 1)

        #augmentation
        # augmentation = Augmentation(float(self.aug_params[0]),
        #                             float(self.aug_params[1]),
        #                             float(self.aug_params[2]),
        #                             float(self.aug_params[3]))
        # view1, view2 = augmentation._feature_masking(batch)


        return Batch(x=x, edge_x=edge_feats, y=y, adj_t=adj, center_mask=center_mask)

class Augmentation:

    def __init__(self, p_f1 = 0.2, p_f2 = 0.1, p_e1 = 0.2, p_e2 = 0.3):
        """
        two simple graph augmentation functions --> "Node feature masking" and "Edge masking"
        Random binary node feature mask following Bernoulli distribution with parameter p_f
        Random binary edge mask following Bernoulli distribution with parameter p_e
        """
        self.p_f1 = p_f1
        self.p_f2 = p_f2
        self.p_e1 = p_e1
        self.p_e2 = p_e2
        # self.method = "BGRL"
    
    def _feature_masking(self, batch):
        device = batch.x.device
        feat_mask1 = torch.FloatTensor(batch.x.shape[1]).uniform_() > self.p_f1
        feat_mask2 = torch.FloatTensor(batch.x.shape[1]).uniform_() > self.p_f2
        feat_mask1, feat_mask2 = feat_mask1.to(device), feat_mask2.to(device)
        x1, x2 = batch.x.clone(), batch.x.clone()
        x1, x2 = x1 * feat_mask1, x2 * feat_mask2

        adj_t1, edge_attr1 = dropout_adj(batch.adj_t, batch.edge_x, p = self.p_e1)
        adj_t2, edge_attr2 = dropout_adj(batch.adj_t, batch.edge_x, p = self.p_e2)

        return (x1, edge_attr1, adj_t1), (x2, edge_attr2, adj_t2)

    def __call__(self, data):
        
        return self._feature_masking(data)


class MPNN(MessagePassing):
    def __init__(self, node_hidden_channels: int, node_out_channels: int,
                 edge_hidden_channels: int, edge_out_channels: int) -> None:
        super(MPNN, self).__init__()
        self.edge_linear = Linear(edge_hidden_channels, edge_hidden_channels)
        self.node_linear_r = Linear(node_hidden_channels, node_hidden_channels)
        self.node_linear_j = Linear(node_hidden_channels, node_hidden_channels)
        self.nn = Linear(node_hidden_channels, node_out_channels)

    def forward(self, x, edge_index, edge_attr):
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index,
                              x=x, edge_attr=edge_attr)
                            #   size=(x.size(0), x.size(0))
        x_r = x[1]
        if x_r is not None:
            x_r = self.node_linear_r(x_r)
            out = out + x_r

        return self.nn(out)
        # return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.edge_linear is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.edge_linear is not None:
            edge_attr = self.edge_linear(edge_attr)

        if self.node_linear_j is not None:
            x_j = self.node_linear_j(x_j)

        return x_j + edge_attr


class Encoder(nn.Module):

    def __init__(self, layer_config, dropout=None, project=False,  num_mpnn=4, **kwargs):
        super().__init__()
        self.num_mpnn = num_mpnn

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()
        for ii in range(num_mpnn):
            if ii < num_mpnn - 1:
                self.conv_layers.append(GINEConv(Linear(layer_config[1], layer_config[1]),edge_dim=layer_config[1]))
                self.bn_layers.append(nn.LayerNorm(layer_config[1]))
                self.act_layers.append(nn.ReLU())
            else:
                self.conv_layers.append(GINEConv(Linear(layer_config[1], layer_config[2]),edge_dim=layer_config[1]))
                self.bn_layers.append(nn.LayerNorm(layer_config[2]))
                self.act_layers.append(nn.ReLU())

        # # self.conv1 = GCNConv(layer_config[0], layer_config[1])
        # # self.conv1 = MPNN(layer_config[1], layer_config[1], layer_config[1], layer_config[1])
        # self.conv1 = GINEConv(Linear(layer_config[1], layer_config[1]),edge_dim=layer_config[1])
        # self.bn1 = nn.BatchNorm1d(layer_config[1], momentum = 0.01)
        # self.prelu1 = nn.ReLU()
        # # self.conv2 = GCNConv(layer_config[1],layer_config[2])
        # # self.conv2 = MPNN(layer_config[1], layer_config[2], layer_config[1], layer_config[2])
        # self.conv2 = GINEConv(Linear(layer_config[1], layer_config[2]),edge_dim=layer_config[1])
        # self.bn2 = nn.BatchNorm1d(layer_config[2], momentum = 0.01)
        # self.prelu2 = nn.ReLU()

    def forward(self, x, edge_index, edge_attr=None):
        for ii in range(self.num_mpnn):
            x = self.conv_layers[ii](x, edge_index, edge_attr=edge_attr)
            x = self.act_layers[ii](self.bn_layers[ii](x))

        # x = self.conv1(x, edge_index, edge_attr=edge_attr)
        # x = self.prelu1(self.bn1(x))
        # x = self.conv2(x, edge_index, edge_attr=edge_attr)
        # x = self.prelu2(self.bn2(x))

        return x


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class BGRL(nn.Module):

    def __init__(self, layer_config, pred_hid, num_mpnn_layers=4, dropout=0.0, moving_average_decay=0.99, epochs=1000, **kwargs):
        super().__init__()
        # self.student_encoder = Encoder(layer_config=layer_config, dropout=dropout, **kwargs)
        self.student_encoder = Encoder(layer_config, num_mpnn=num_mpnn_layers)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(moving_average_decay, epochs)
        rep_dim = layer_config[-1]
        # self.student_predictor = nn.Sequential(nn.Linear(rep_dim, pred_hid), nn.ReLU(), nn.Linear(pred_hid, rep_dim))
        self.student_predictor = Sequential(
            Linear(rep_dim, pred_hid),
            LayerNorm(pred_hid),
            ReLU(inplace=True),
            Linear(pred_hid, rep_dim),
        )
        self.student_predictor.apply(init_weights)
    
    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def forward(self, x1, x2, edge_index_v1, edge_index_v2, edge_weight_v1=None, edge_weight_v2=None):
        v1_student = self.student_encoder(x=x1, edge_index=edge_index_v1, edge_attr=edge_weight_v1)
        v2_student = self.student_encoder(x=x2, edge_index=edge_index_v2, edge_attr=edge_weight_v2)

        v1_pred = self.student_predictor(v1_student)
        v2_pred = self.student_predictor(v2_student)
        
        with torch.no_grad():
            v1_teacher = self.teacher_encoder(x=x1, edge_index=edge_index_v1, edge_attr=edge_weight_v1)
            v2_teacher = self.teacher_encoder(x=x2, edge_index=edge_index_v2, edge_attr=edge_weight_v2)

        return v1_pred, v2_pred, v1_teacher, v2_teacher


        # loss1 = loss_fn(v1_pred, v2_teacher.detach())
        # loss2 = loss_fn(v2_pred, v1_teacher.detach())

        # loss = loss1 + loss2
        # return v1_student, v2_student, loss.mean()


class DPMNN(LightningModule):
    def __init__(self, node_in_channels: int, node_out_channels: int,
                 bgrl_in: int, bgrl_hid: int, bgrl_out: int,
                 pre_hid: int,
                 edge_in: int, edge_hid: int, edge_out: int,
                 num_relations: int, num_mpnn_layers: int, 
                 dropout: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        # self.model = model.lower()
        self.num_relations = num_relations
        self.dropout = dropout

        # self.node_trans = Linear(in_channels, hidden_channels)
        # self.edge_trans = Linear(edge_in_channels, edge_hidden_channels)
        self.node_trans = Sequential(
            Linear(node_in_channels, bgrl_in),
            LayerNorm(bgrl_in),
            ReLU(inplace=True),
            # Dropout(p=self.dropout)
        )
        self.edge_trans = Sequential(
            Linear(edge_in, bgrl_in),
            LayerNorm(bgrl_in),
            ReLU(inplace=True),
            # Dropout(p=self.dropout)
        )

        # layers = [self._dataset.x.shape[1]] + hidden_layers
        layers_config = [bgrl_in] + [bgrl_hid] + [bgrl_out]
        self.bgrl =BGRL(layer_config=layers_config, 
                        pred_hid=pre_hid, 
                        use_bn=True,
                        num_mpnn_layers=num_mpnn_layers
        )

        self.pre_logits = Sequential(
            Linear(bgrl_out, node_out_channels),
            LayerNorm(node_out_channels),
            ReLU(inplace=True),
        )

        # self.dp = MPNN(num_relations, dropout=args.dropout, epochs=args.epochs)

        # self.mpnn = MPNN(in_channels, hidden_channels, edge_in_channels, edge_hidden_channels)

        # self.student_encoder = MPNN(layer_config[0], layer_config[1], layer_config[2], 7, layer_config[1])
        # self.teacher_encoder = copy.deepcopy(self.student_encoder)

        # self.convs = ModuleList()
        # self.norms = ModuleList()
        # self.skips = ModuleList()


        # for _ in range(num_layers):
        #     self.norms.append(BatchNorm1d(hidden_channels))

        # self.skips.append(Linear(in_channels, hidden_channels))
        # for _ in range(num_layers - 1):
        #     self.skips.append(Linear(hidden_channels, hidden_channels))

        # self.mlp = Sequential(
        #     Linear(hidden_channels, hidden_channels),
        #     BatchNorm1d(hidden_channels),
        #     ReLU(inplace=True),
        #     Dropout(p=self.dropout),
        #     Linear(hidden_channels, out_channels),
        # )

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.aug_params = [0.4, 0.4, 0.2, 0.2]

    # def dpmnn(self, graph):
    #     x, edge_x, adj_t = graph
    #     x = self.node_trans(x)
    #     edge_x = self.edge_trans(edge_x)
    #     self.mpnn


    def forward(self, graph=None, au_graph1=None, au_graph2=None, bgrl=True):

        # # x = self.student_encoder(x, edge_x, adj_t)
        # x = self.student_encoder(x, edge_x, adj_t)
        # x = self.teacher_encoder(x, edge_x, adj_t)
        # # x = self.student_encoder(x, edge_x, adj_t)
        # x = self.skips[0](x)

        if bgrl:
            x1, edge_x1, adj_t1 = au_graph1
            x2, edge_x2, adj_t2 = au_graph2

            x1 = self.node_trans(x1)
            x2 = self.node_trans(x2)
            edge_x1 = self.edge_trans(edge_x1)
            edge_x2 = self.edge_trans(edge_x2)

            # v1_output, v2_output, loss = self.bgrl(
            #     x1=x1, x2=x2, edge_index_v1=adj_t1, edge_index_v2=adj_t2,
            #     edge_weight_v1=edge_x1, edge_weight_v2=edge_x2)
            # return loss

            v1_pred, v2_pred, v1_teacher, v2_teacher = self.bgrl(
                x1=x1, x2=x2, edge_index_v1=adj_t1, edge_index_v2=adj_t2,
                edge_weight_v1=edge_x1, edge_weight_v2=edge_x2)
            return v1_pred, v2_pred, v1_teacher, v2_teacher

        else:
            x, edge_x, adj_t = graph
            with torch.no_grad():
                x = self.node_trans(x)
                edge_x = self.edge_trans(edge_x)
                emb = self.bgrl.student_encoder(x=x, edge_index=adj_t, edge_attr=edge_x)
                emb = self.bgrl.student_predictor(emb)
            emb = self.pre_logits(emb)
            
            return emb

        
        # x = self.node_trans(x)
        # edge_x = self.edge_trans(x)

        # for i, adj_t in enumerate(adjs_t):
        #     x_target = x[:adj_t.size(0)]

        #     # out = self.skips[i](x_target)
        #     # for j in range(self.num_relations):
        #     #     edge_type = adj_t.storage.value() == j
        #     #     subadj_t = adj_t.masked_select_nnz(edge_type, layout='coo')
        #     #     subadj_t = subadj_t.set_value(None, layout=None)
        #     #     if subadj_t.nnz() > 0:
        #     #         out += self.convs[i][j]((x, x_target), subadj_t)

        #     x = self.norms[i](out)
        #     # x = F.elu(x) if self.model == 'rgat' else F.relu(x)
        #     x = F.dropout(x, p=self.dropout, training=self.training)

        # return self.mlp(x)

    def training_step(self, batch, batch_idx: int):
        self.bgrl.update_moving_average()
        augmentation = Augmentation(float(self.aug_params[0]),
                                    float(self.aug_params[1]),
                                    float(self.aug_params[2]),
                                    float(self.aug_params[3]))
        view1, view2 = augmentation._feature_masking(batch)

        v1_pred, v2_pred, v1_teacher, v2_teacher = self(au_graph1=(view1[0], view1[1], view1[2]),
                               au_graph2=(view2[0], view2[1], view2[2]))
        bgrl_loss1 = loss_fn(v1_pred[batch.center_mask], v2_teacher.detach()[batch.center_mask])
        bgrl_loss2 = loss_fn(v2_pred[batch.center_mask], v1_teacher.detach()[batch.center_mask])
        train_loss_bgrl = (bgrl_loss1 + bgrl_loss2).mean()    

        y_hat = self(graph=(batch.x, batch.edge_x, batch.adj_t),bgrl=False)
        train_loss_sup = F.cross_entropy(y_hat[batch.center_mask], batch.y)
        self.train_acc(y_hat.softmax(dim=-1)[batch.center_mask], batch.y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return train_loss_sup + train_loss_bgrl

    def validation_step(self, batch, batch_idx: int):
        y_hat = self((batch.x, batch.edge_x, batch.adj_t),bgrl=False)
        self.val_acc(y_hat.softmax(dim=-1)[batch.center_mask], batch.y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        self.test_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--mpnn_layers', type=int, default=4)
    parser.add_argument('--model', type=str, default='dpmnn',
                        choices=['rgat', 'rgraphsage', 'dpmnn'])
    parser.add_argument('--sizes', type=str, default='60-40')
    # parser.add_argument('--in-memory', action='store_true')
    parser.add_argument('--in-memory', type=bool, default=True)
    parser.add_argument('--device', type=str, default='0,')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    print(args)

    seed_everything(42)
    datamodule = MAG240M(ROOT, args.batch_size, args.sizes, args.in_memory)

    if not args.evaluate:
        # model = DPMNN(datamodule.num_features,
        #              datamodule.num_classes, args.hidden_channels,
        #              datamodule.num_relations, num_layers=len(args.sizes),
        #              dropout=args.dropout)
        model = DPMNN(datamodule.num_features, datamodule.num_classes,
                 args.hidden_channels, args.hidden_channels, args.emb_dim,
                 args.hidden_channels, 
                 7, args.hidden_channels, args.hidden_channels, 
                 datamodule.num_relations, num_mpnn_layers=args.mpnn_layers, dropout=args.dropout)
        print(f'#Params {sum([p.numel() for p in model.parameters()])}')
        checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max',
                                              save_top_k=1)
        trainer = Trainer(gpus=args.device, max_epochs=args.epochs,
                          callbacks=[checkpoint_callback],
                          default_root_dir=f'logs/subg/{args.model}')
        trainer.fit(model, datamodule=datamodule)

    if args.evaluate:
        dirs = glob.glob(f'logs/subg/{args.model}/lightning_logs/*')
        version = max([int(x.split(os.sep)[-1].split('_')[-1]) for x in dirs])
        logdir = f'logs/subg/{args.model}/lightning_logs/version_{version}'
        print(f'Evaluating saved model in {logdir}...')
        ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]

        trainer = Trainer(gpus=args.device, resume_from_checkpoint=ckpt)
        model = RGNN.load_from_checkpoint(
            checkpoint_path=ckpt, hparams_file=f'{logdir}/hparams.yaml')

        datamodule.batch_size = 16
        datamodule.sizes = [160] * len(args.sizes)  # (Almost) no sampling...

        trainer.test(model=model, datamodule=datamodule)

        evaluator = MAG240MEvaluator()
        loader = datamodule.hidden_test_dataloader()

        model.eval()
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        y_preds = []
        for batch in tqdm(loader):
            batch = batch.to(device)
            with torch.no_grad():
                out = model(batch.x, batch.adjs_t).argmax(dim=-1).cpu()
                y_preds.append(out)
        res = {'y_pred': torch.cat(y_preds, dim=0)}
        evaluator.save_test_submission(res, f'results/{args.model}',
                                       mode='test-dev')
