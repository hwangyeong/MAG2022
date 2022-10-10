
from array import array
from cgi import test
import torch
import os
import numpy as np
import networkx as nx
import scipy.sparse as sp
from ogb.lsc import MAG240MDataset
from torch_geometric.utils import subgraph, to_networkx
from torch_geometric.data import HeteroData
from torch_geometric.utils.mask import index_to_mask, mask_to_index

np.random.seed(1234)
# dataset = MAG240MDataset(root="dataset_path")
# edge_index = dataset.edge_index('author', 'writes', 'paper')
# # edge_index = dataset.edge_index('author', 'writes', 'paper')
# path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
# aa = torch.load(path).csr()

# print(dataset.num_papers) # number of paper nodes
# print(dataset.num_authors) # number of author nodes
# print(dataset.num_institutions) # number of institution nodes
# print(dataset.num_paper_features) # dimensionality of paper features
# print(dataset.num_classes) # number of subject area classes

# print(dataset.paper_label) # numpy array of shape (num_papers, ), storing target labels of papers.

# print(dataset.num_papers + dataset.num_authors + dataset.num_institutions)

def find_intersection_set(tensor1 ,tensor2, num_all_papers=121751666):
    a1 = index_to_mask(tensor1, size=num_all_papers)
    a2 = index_to_mask(tensor2, size=num_all_papers)
    intersection = a1 & a2
    return intersection
class ExtractSubgraph():

    def __init__(self, root) -> None:
        self.root = root
        self.dataset = MAG240MDataset(root)

        self.selected_papers, self.lables = None, None
        self.subg = None
        self.num_classes = None
        self.train_mask, self.valid_mask = None, None
    
    def select_seeds(self, num_seed): # 选验证集的文章作为起始种子
        valid_idx = self.dataset.get_idx_split('valid')
        # num_seed = round(train_idx.shape[0] * select_ratio)
        seed_papers = torch.from_numpy(np.random.choice(valid_idx, num_seed, replace=False))
        return seed_papers

    def select_papers(self, seed_papers):
        selected_valid_papers = seed_papers
        path = f'{self.dataset.dir}/paper_to_paper_symmetric.pt'
        indptr, indices, _ = torch.load(path).csr()

        def explore_papers(seed_papers, indptr, indices):
            neis_start = indptr[seed_papers]
            neis_end = indptr[seed_papers + 1]
            selected_papersid = [torch.arange(neis_start[ii], neis_end[ii]) for ii in range(seed_papers.shape[0])]
            selected_papersid = torch.cat(selected_papersid)
            # selected_papers = torch.unique(selected_papersid)
            selected_papers = indices[selected_papersid]
            selected_papers = torch.cat((selected_papers, seed_papers)) # 拼上seed
            selected_papers = torch.unique(selected_papers).long()
            return selected_papers

        selected_papers_from_valid = explore_papers(seed_papers, indptr, indices) # 先从验证集作为种子找一轮
        train_idx = torch.from_numpy(self.dataset.get_idx_split('train'))
        selected_train_papers = find_intersection_set(train_idx, selected_papers_from_valid) # bool张量 (121751666)
        selected_train_papers = mask_to_index(selected_train_papers)
        selected_papers_from_train = explore_papers(selected_train_papers, indptr, indices) # 再从第一次筛的结果中，训练的paper作为种子找一轮
        selected_papers = torch.cat((selected_papers_from_train, selected_papers_from_valid)) # 合并两次的结果
        selected_papers = torch.unique(selected_papers).long()

        # 统计arxiv中的paper在选中paper中的情况
        # train_idx = torch.from_numpy(self.dataset.get_idx_split('train'))
        # valid_idx = torch.from_numpy(self.dataset.get_idx_split('valid'))
        test_idx = torch.from_numpy(self.dataset.get_idx_split('test'))
        selected_train_papers = find_intersection_set(train_idx, selected_papers) #bool张量 (121751666)
        # selected_valid_papers = find_intersection_set(valid_idx, selected_papers) #bool张量 (121751666)
        selected_test_papers = find_intersection_set(test_idx, selected_papers) #bool张量 (121751666)

        print('训练集文章 {} 篇, 验证集文章 {} 篇, 测试集文章 {} 篇'.format(
            torch.numel(selected_train_papers[selected_train_papers]),
            # torch.numel(selected_valid_papers[selected_valid_papers]),
            selected_valid_papers.shape[0],
            torch.numel(selected_test_papers[selected_test_papers])
        ))
        print('文章总数 {} 篇'.format(torch.numel(selected_papers)))

        # 整理train_mask和valid_mask
        selected_train_papers = mask_to_index(selected_train_papers)
        self.train_mask = torch.from_numpy(np.in1d(selected_papers.numpy(), selected_train_papers.numpy())) # 保留train_mask, 后续充新编号过之后再调取
        self.valid_mask = torch.from_numpy(np.in1d(selected_papers.numpy(), selected_valid_papers.numpy()))

        # years = torch.from_numpy(self.dataset.all_paper_year)
        # selected_papers_year = years[selected_papers]

        # labels
        y = torch.from_numpy(self.dataset.all_paper_label) # (121751666)
        
        # self.split_train = selected_train_papers
        # self.split_valid = selected_valid_papers
        need_new_label_papers = torch.cat((selected_train_papers, selected_valid_papers)) # 需要新标签的点
        #(选中paper中属于原数据train的papre，以及选定的valid)

        # ori_labels = y[selected_papers]
        keep_labels = y[need_new_label_papers].long() # 不是全部标签都要,记录需要保留label的点
        keep_labels = torch.unique(keep_labels)
        num_labels = keep_labels.shape[0] #需要保留标签的个数
        self.num_classes = num_labels

        ori_label_new_label = torch.zeros(153, dtype=torch.float64) - 1 # 新老标签对应关系 
        ori_label_new_label[keep_labels] = torch.arange(num_labels, dtype=torch.float64) #原标签为float64类型，保持一致
        
        y_new = y.new_full((y.shape), float('nan')) #更新原数据的label,只给需要标签的点加label
        y_new[need_new_label_papers] = ori_label_new_label[y[need_new_label_papers].long()] #更新原数据的label,只给需要标签的点加labe
        new_label = y_new[selected_papers]

        # # new_label = y.new_full((selected_papers.shape), float('nan')) 
        # new_label = ori_label_new_label[y[need_new_label_papers].long()]

        # year
        years = torch.from_numpy(self.dataset.all_paper_year)
        selected_year = years[selected_papers]

        return selected_papers, new_label, selected_year

    def select_other_entity(self, given_entities, edge_type, reverse_edge=False): # 选异质边的邻居，用given_entities找邻居
        if edge_type == 'ap':
            edge_index = self.dataset.edge_index('author', 'writes', 'paper')
            shape_ = (self.dataset.num_papers, self.dataset.num_authors)
        elif edge_type == 'ai':
            edge_index = self.dataset.edge_index('author', 'institution')
            shape_ = (self.dataset.num_authors, self.dataset.num_institutions)
        if reverse_edge:
            edge_index = edge_index[[1,0]] # p-a
        csr = sp.csr_matrix((np.ones(edge_index.shape[1]),(edge_index[0], edge_index[1])), 
                        shape=shape_)
        indptr, indices = csr.indptr, csr.indices
        neis_start = indptr[given_entities]
        neis_end = indptr[given_entities + 1]
        selected_ids = [torch.arange(neis_start[ii], neis_end[ii]) for ii in range(given_entities.shape[0])]
        selected_ids = torch.cat(selected_ids)
        selected_entities = indices[selected_ids]
        selected_entities = np.unique(selected_entities)
        return torch.from_numpy(selected_entities).long()

    def select_edges_pp(self, selected_nodes, edge_type):
        if edge_type == 'pp':
            edge_index = self.dataset.edge_index('paper', 'cites', 'paper')
            edge_name = 'paper___cites___paper'

        edge_index = torch.from_numpy(edge_index)
        subg,_ = subgraph(selected_nodes, edge_index)
        path = os.path.join(self.root, 'mag_sub', 'sub_1e-4', edge_name, 'edge_index.npy')
        np.save(path, edge_index)
        return subg

    # def select_edges_ap(self, selected_authors, selected_papers):
    #     edge_index = self.dataset.edge_index('author', 'writes', 'paper')
    #     edge_name = 'author___writes___paper'
    #     edge_index[1] += self.dataset.num_papers
    #     edge_index = torch.from_numpy(edge_index)
    #     selected_nodes = torch.cat(selected_authors. selected_papers)
    #     subg,_ = subgraph(selected_nodes, edge_index)


        # elif edge_type == 'ai':
        #     edge_index = self.dataset.edge_index('author', 'institution')
        #     edge_name = 'author___affiliated_with___institution'

    def select_edges(self, selected_papers, selected_authors, selected_institutions):

        heter_g = HeteroData()
        x = np.memmap('/data/zhaohuanjing/mag/dataset_path/mag240m_kddcup2021/full_feat.npy', dtype=np.float16,
                      mode='r', shape=(244160499, 768))
        # all_feat = torch.from_numpy(x)
        heter_g['paper'].x = torch.zeros(self.dataset.num_papers, 1)
        heter_g['author'].x = torch.zeros(self.dataset.num_authors, 1)
        heter_g['institution'].x = torch.zeros(self.dataset.num_institutions, 1)
        # heter_g['paper'].x = all_feat[self.dataset.num_papers]
        # heter_g['author'].x = all_feat[self.dataset.num_papers: self.dataset.num_papers + self.dataset.num_authors]
        # heter_g['institution'].x = all_feat[self.dataset.num_papers + self.dataset.num_authors: 244160499]
        heter_g['author', 'writes', 'paper'].edge_index = torch.from_numpy(self.dataset.edge_index('author', 'writes', 'paper'))
        heter_g['paper', 'citees', 'paper'].edge_index = torch.from_numpy(self.dataset.edge_index('paper', 'cites', 'paper'))
        heter_g['author', 'affiliated', 'institution'].edge_index = torch.from_numpy(self.dataset.edge_index('author', 'institution'))
        # dic_selected = {
        #     'paper': selected_papers,
        #     'author': selected_authors,
        #     'institution': selected_institution
        # }
        # subg = heter_g.subgraph(dic_selected)
        from torch_geometric.utils import bipartite_subgraph
        subset_dict = {'paper':selected_papers, 'author':selected_authors, 'institution':selected_institutions}
        subg = HeteroData()
        for edge_type in heter_g.edge_types:
            src, _, dst = edge_type
            if src not in subset_dict or dst not in subset_dict:
                continue
            edge_index, _, edge_mask = bipartite_subgraph(
                (subset_dict[src], subset_dict[dst]),
                heter_g[edge_type].edge_index,
                relabel_nodes=False,
                size=(heter_g[src].num_nodes, heter_g[dst].num_nodes),
                return_edge_mask=True,
            )
            subg[edge_type].edge_index = edge_index

        dic_nodetype_nodenum = {
            'paper': self.dataset.num_papers,
            'author': self.dataset.num_authors,
            'institution': self.dataset.num_institutions
        }

        dic_select_features = {
            'paper': x[selected_papers],
            'author': x[selected_authors + self.dataset.num_papers],
            'institution': x[selected_institutions + self.dataset.num_papers + self.dataset.num_authors]
        }
        

        dic_relabel_features = {}
        dic_relabel_edgeindex = {}
        # relabel
        for edge_type in subg.edge_types:
            src, _, dst = edge_type
            # num_src = subg[src].x.shape[0] #该类型节点选了多少个
            # num_dst = subg[dst].x.shape[0]

            node_idx_i = torch.zeros(dic_nodetype_nodenum[src], dtype=torch.long)
            node_idx_j = torch.zeros(dic_nodetype_nodenum[dst], dtype=torch.long)

            edge_index = subg[edge_type]['edge_index']

            # node_idx_i[edge_index[0]] = torch.arange(num_src)
            # node_idx_j[edge_index[1]] = torch.arange(num_dst)
            
            selected_i = subset_dict[src]
            selected_j = subset_dict[dst]

            node_idx_i[selected_i] = torch.arange(selected_i.shape[0]) # N
            node_idx_j[selected_j] = torch.arange(selected_j.shape[0]) # N

            # node_idx_i = torch.zeros(node_mask[0].size(0), dtype=torch.long,
            #                         device=device)
            # node_idx_j = torch.zeros(node_mask[1].size(0), dtype=torch.long,
            #                         device=device)
            # node_idx_i[node_mask[0]] = torch.arange(node_mask[0].sum().item(),
            #                                         device=device)
            # node_idx_j[node_mask[1]] = torch.arange(node_mask[1].sum().item(),
            #                                         device=device)

            edge_index = torch.stack(
                [node_idx_i[edge_index[0]], node_idx_j[edge_index[1]]])

            dic_relabel_edgeindex[edge_type] = edge_index

            # extract features
            if src not in dic_relabel_features:
                feature_idx_i = node_idx_i[selected_i]
                dic_relabel_features[src] = dic_select_features[src][feature_idx_i]
            if dst not in dic_relabel_features:
                feature_idx_j = node_idx_j[selected_j]
                dic_relabel_features[dst] = dic_select_features[dst][feature_idx_j]
        
        subg_re = HeteroData()
        subg_re['paper'].x = dic_relabel_features['paper']
        subg_re['paper'].num_nodes = dic_relabel_features['paper'].shape[0]
        subg_re['author'].x = dic_relabel_features['author']
        subg_re['author'].num_nodes = dic_relabel_features['author'].shape[0]
        subg_re['institution'].x = dic_relabel_features['institution']
        subg_re['institution'].num_nodes = dic_relabel_features['institution'].shape[0]
        subg_re['author', 'writes', 'paper'].edge_index = dic_relabel_edgeindex[('author', 'writes', 'paper')]
        subg_re['paper', 'citees', 'paper'].edge_index = dic_relabel_edgeindex[('paper', 'citees', 'paper')]
        subg_re['author', 'affiliated', 'institution'].edge_index = dic_relabel_edgeindex[('author', 'affiliated', 'institution')]
        return subg_re
        

    def extract(self, num_valid):
        seed_papers = self.select_seeds(num_valid)

        self.selected_papers, self.labels, self.selected_years = self.select_papers(seed_papers)
        # self.sub_pp = self.select_edges_pp(selected_papers, 'pp')
        
        selected_authors = self.select_other_entity(self.selected_papers, 'ap', reverse_edge=True)
        # selected_authors += self.dataset.num_papers # 为抽边准备
        selected_institution = self.select_other_entity(selected_authors, 'ai', reverse_edge=False)

        self.subg = self.select_edges(self.selected_papers, selected_authors, selected_institution)
        print(self.subg)
    
    # def save_features(self):
    #     s


    def save_subg(self, papaer_feat_only):
        path = os.path.join(self.root, 'mag_sub', 'sub_1e-4', 'mag240m_kddcup2021')
        dic_meta = {
            'paper': self.subg['paper'].x.shape[0],
            'author': self.subg['author'].x.shape[0],
            'institution': self.subg['institution'].x.shape[0],
            'num_classes': self.num_classes
            }
        meta_path = os.path.join(path, 'meta.pt')
        torch.save(dic_meta, meta_path)

        dic_split = {
            'train': torch.arange(self.subg['paper'].x.shape[0])[self.train_mask].numpy(),
            'valid': torch.arange(self.subg['paper'].x.shape[0])[self.valid_mask].numpy(),
            'test': np.array([], dtype=np.int64),
        }
        split_path = os.path.join(path, 'split_dict.pt')
        torch.save(dic_split, split_path)

        for edge, edge_index_dic in self.subg.edge_items():
            if edge == ('author', 'writes', 'paper'):
                edge_name = 'author___writes___paper'
            elif edge == ('paper', 'citees', 'paper'):
                edge_name = 'paper___cites___paper'
            elif edge == ('author', 'affiliated', 'institution'):
                edge_name = 'author___affiliated_with___institution'
            edge_index = edge_index_dic['edge_index']
            path = os.path.join(self.root, 'mag_sub', 'sub_1e-4', 'mag240m_kddcup2021/processed', edge_name)
            # os.mkdir(path)
            path = os.path.join(path, 'edge_index.npy')
            np.save(path, edge_index)

        path = os.path.join(self.root, 'mag_sub', 'sub_1e-4', 'mag240m_kddcup2021/processed/paper')
        # os.mkdir(path)
        # label
        path_label = os.path.join(path, 'node_label.npy')
        np.save(path_label, self.labels)
        # year
        path_year = os.path.join(path, 'node_year.npy')
        np.save(path_year, self.selected_years)

        # feature
        if papaer_feat_only:
            path_feature = os.path.join(path, 'node_feat.npy')
            paper_feat = self.subg['paper'].x
            np.save(path_feature, paper_feat)
        else:
            path_feature = os.path.join(path, 'all_feat.npy')
            paper_feat = self.subg['paper'].x
            author_feat = self.subg['author'].x
            institution_feat = self.subg['institution'].x
            feat_all = np.concatenate((paper_feat,author_feat,institution_feat), axis=0)
            np.save(path_feature, feat_all)

        
        # N = dataset.num_papers + dataset.num_authors + dataset.num_institutions

        # x = np.memmap(f'{dataset.dir}/full_feat.npy', dtype=np.float16,
        #               mode='r', shape=(N, self.num_features))

    def check_subg(self):
        print('文章数 {}, 作者数 {}, 机构数 {}'.format(
            self.subg['paper'].x.shape[0],
            self.subg['author'].x.shape[0],
            self.subg['institution'].x.shape[0]),
        )
        nxg = to_networkx(self.subg.to_homogeneous()).to_undirected()
        check = list(nx.algorithms.connected_components(nxg))
        if len(check) == 1:
            print('全连通图')
        else:
            print('{} 个连通子图'.format(len(check)))
        


if __name__ == '__main__':
    exs = ExtractSubgraph('dataset_path')

    # exs.select_seeds(1000)
    # exs.select_papers(ps)

    exs.extract(1000)
# #     exs.select_seeds(0.0001)
    exs.check_subg()

    # exs.extract(1e-4)
    exs.save_subg(papaer_feat_only=True)

    print()