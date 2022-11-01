
import dgl
from matplotlib.cbook import print_cycles
import torch
import os
import numpy as np
import tqdm
from ogb.lsc import MAG240MDataset
import multiprocessing


filepath = "/data/zhaohuanjing/mag/dataset_path/mplp_data/preprocess"
graph_name = "dgl_graph_full_heterogeneous.bin"
root = "dataset_path"
dataset  = MAG240MDataset(root)

graph_path = os.path.join(filepath, graph_name)
# graph = dgl.load_graphs(graph_path)[0][0]

# # build pp cite graph bi-direction
# # pcp = graph['cites'].edges()
# # pcbp = graph['cited_by'].edges()

# # print('begin cat')
# # p_src = torch.cat([pcp[0], pcbp[0]])
# # p_dst = torch.cat([pcp[1], pcbp[1]])

# # # del graph
# # print('cat finish')
# # ppgraph_cite = dgl.graph((p_src, p_dst), num_nodes=dataset.num_papers)

# # print("saving")
# # dgl.save_graphs(os.path.join(filepath, 'ppgraph_cite.bin'), ppgraph_cite)



# # build pp with coauthor
# pa = graph['writed_by'].edges()
# # pa = graph['writes'].edges()

# g_pa = dgl.heterograph({
#     ('paper', 'writed_by', 'author'): (pa[0], pa[1]),
#     ('author', 'writes', 'paper'): (pa[1], pa[0])
# })

# labels = np.load(
#     os.path.join(root, "mag240m_kddcup2021", "processed", "paper",
#                  "node_label.npy"),
#     mmap_mode="r")

# def find_similar_paper_single(pid):
#     authors = g_pa.successors(pid, etype='writed_by')
#     _, papers = g_pa.out_edges(authors, etype='writes')
#     papers, counts = np.unique(papers, return_counts=True)
#     # papers, counts = np.unique(
#     #     g_pa.successors(authors, etype='writes'),
#     #     return_counts=True)
#     indegree = g_pa.out_degrees(papers, etype='writed_by')
#     indegree = indegree + g_pa.out_degrees(pid, etype='writed_by') - counts
#     indegree = indegree.numpy()
#     counts = counts / indegree
#     # select neighbor with following condition
#     #   1. coauthor jaccard  > 0.5 
#     #   2. labels are not NaN
#     mask = (counts >= 0.5) & (papers != pid) & (
#         ~np.isnan(labels[papers.tolist()]))
#     papers = papers[mask]
#     counts = counts[mask]
#     edges = np.vstack([papers, np.ones_like(papers) * pid])
#     return edges.T.astype("int64")

# def find_similar_paper(start_pid):
#     all_edges = []
#     for pid in range(start_pid,
#                      min(dataset.num_papers, start_pid + batch_size)):
#         edges = find_similar_paper_single(pid)
#         if len(edges) > 0:
#             all_edges.append(edges)
#     return np.concatenate(all_edges, dtype="int64")


# batch_size = 5000
# max_workers = 50
# batch_papers = np.arange(0, dataset.num_papers, step=batch_size)
# edges = []
# with multiprocessing.Pool(max_workers) as pool:
#     chunksize = 1000
#     imap_unordered_it = pool.imap_unordered(find_similar_paper, batch_papers,
#                                             chunksize)
#     start = 0
#     for edge_feat in tqdm.tqdm(imap_unordered_it, total=len(batch_papers)):
#         if len(edge_feat) > 0:
#             edges.append(edge_feat.astype("int64"))
#     edges = np.concatenate(edges, dtype="int64")
#     print(edges.shape)
#     # edge_types = np.full([edges.shape[0]], 6, dtype="int32")
#     edges = edges.T
#     ppgraph_co = dgl.graph((edges[0], edges[1]),
#         num_nodes=dataset.num_papers
#         )
#     print("saving")
#     dgl.save_graphs(os.path.join(filepath, "paper_coauthor_paper_symmetric_jc0.5.bin"), ppgraph_co)

# # load graph from PGL
# import pgl
# ppgraph_cojc = pgl.Graph.load(
#     os.path.join(root, "mag240m_kddcup2021", "paper_coauthor_paper_symmetric_jc0.5"))

# edges = ppgraph_cojc.edges.T
# ppgraph_co = dgl.graph((edges[0], edges[1]),
#         num_nodes=dataset.num_papers
#         )
# dgl.save_graphs(os.path.join(filepath, "paper_coauthor_paper_symmetric_jc0.5_pgl.bin"), ppgraph_co)        





print('DONE')