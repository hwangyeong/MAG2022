
import torch
from torch_geometric.utils import subgraph, bipartite_subgraph
# ap_edges = torch.LongTensor([[0,0,1,1,1],[1,2,0,3,4]])
# bipartite_subgraph((torch.tensor([1]), torch.tensor([0,4])), ap_edges, return_edge_mask=True, relabel_nodes=True)

from ogb.lsc import MAG240MDataset

dataset = MAG240MDataset(root='dataset_path')
# print(dataset.num_authors)
year = torch.from_numpy(dataset.all_paper_year)
print(torch.numel(year[year>2019]))

y = torch.from_numpy(dataset.all_paper_label)
aa = y.nan_to_num(-2)
print(aa.max()+1) #153
print()