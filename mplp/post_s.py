# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import paddle
from tqdm import tqdm
import dgl
import numpy as np
import torch
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
import time
import sys
import os
import logging
import glob


# dataset_path, seed_path, graph_path = sys.argv[1:4]
# root = "dataset_path"

dataset_path = "/data/zhaohuanjing/mag/dataset_path"

seed_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/mplp/outputs/mplp_m2v"
# graph_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/preprocess/ppgraph_cite.bin"
graph_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/preprocess/paper_coauthor_paper_symmetric_jc0.5_pgl.bin"
y_re_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/mplp/data/y_base.npy"

logging.basicConfig(
     format= '%(asctime)s %(levelname)s %(module)s : %(message)s',
     level=logging.INFO
)
logger = logging.getLogger(__name__)

dataset  = MAG240MDataset(dataset_path)
evaluator = MAG240MEvaluator()
# general y_pred_valid_all first

print("\nEvaluating at validation dataset...")
# y_pred_valid_all = []
acc_all = []
y_pred_v = []
idx_v = []
y_true_valid = np.load(y_re_path)
for fpath in glob.glob(os.path.join(seed_path, 'cv-*')):
    print("Loading predictions from %s" % fpath)
    y = torch.as_tensor(np.load(os.path.join(fpath, "y_pred_valid.npy"))).softmax(axis=1).numpy()
    y_pred_v.append(y)
    idx = np.load(os.path.join(fpath, "idx_valid.npy"))
    idx_v.append(idx)
idx_v = np.concatenate(idx_v, axis=0)
y_pred_v = np.concatenate(y_pred_v, axis=0)
y_true_v = y_true_valid[idx_v]
# y_pred_valid_all.append(y_pred_v[np.argsort(idx_v)])
y_pred_valid_all = y_pred_v[np.argsort(idx_v)]

acc = evaluator.eval(
    {'y_true': y_true_v, 'y_pred': y_pred_v.argmax(axis=1)}
)['acc']
acc_all.append(acc)
print("valid accurate: %.4f" % acc)

np.save(os.path.join(seed_path, 'y_pred_valid_all'), y_pred_valid_all)
print('y_pred_valid_all is done')


alpha = 0.8
cuda_ = 0

device = torch.device("cuda:%s" % cuda_ if torch.cuda.is_available() else "cpu")
# fname = os.path.join(datapath, 'y_base.npy')
# logger.info("Loading %s" % fname)
# labels = np.load(fname)
# labels = dataset.paper_label
labels = np.load(
    os.path.join(dataset_path, "mag240m_kddcup2021", "processed", "paper",
                 "node_label.npy"),
    mmap_mode="r")
split = torch.load(os.path.join(dataset_path, "mag240m_kddcup2021", "split_dict.pt"))

# graph_path = os.path.join(root, "paper_coauthor_paper_symmetric_jc0.5")
# graph = pgl.Graph.load(graph_path)
graph = dgl.load_graphs(graph_path)[0][0]

import numpy as np



# fold_id = int(sys.argv[2])
# model_name = sys.argv[3]
# alpha = float(sys.argv[1])
model_name = seed_path.split('/')[-1]
save_model_name = model_name + "_diff%s" % alpha

for fpath in glob.glob(os.path.join(seed_path, 'cv-*')):
    fold_id = int(fpath.split('-')[-1])
    try:
        os.makedirs(os.path.join(seed_path, save_model_name))
    except Exception as e:
        print(e)
        pass

    valid_label = np.load(os.path.join(seed_path, "y_pred_valid_all.npy"))
    test_label = np.load(os.path.join(fpath, "y_pred_test.npy"))

    # prepare indgree
    indegree = graph.in_degrees()
    indegree = (1.0 / (indegree + 1)).reshape([-1, 1])

    def aggr(batch, y, nxt_y, y0, alpha):
        pred_src, pred_dst = graph.in_edges(batch.numpy())
        self_label = torch.from_numpy(y[batch.numpy()])
        self_label0 = torch.from_numpy(y0[batch.numpy()])
        pred_id = []
        # for n, p in enumerate(pred):
        #     if len(p) > 0:
        #         pred_id.append(np.ones(len(p)) * n)
        for n in range(pred_dst.shape[0]):
            if n == 0:
                pred_id.append(0)
            else:
                if pred_dst[n] == pred_dst[n-1]:
                    pred_id.append(pred_id[n-1])
                else:
                    pred_id.append(pred_id[n-1] + 1)
        # pred_cat = np.concatenate(pred)
        pred_cat = pred_src
        # pred_id_cat = torch.from_numpy(np.concatenate(pred_id, dtype="int64"))
        pred_id_cat = torch.from_numpy(np.array(pred_id, dtype="int64"))
        # pred_cat_pd = torch.from_numpy(pred_cat)
        pred_cat_pd = pred_src

        pred_label = torch.from_numpy(y[pred_cat])

        pred_norm = torch.gather(indegree, 0, pred_cat_pd.unsqueeze(0).T)
        self_norm = torch.gather(indegree, 0, batch.unsqueeze(0).T)

        others = torch.zeros_like(self_label)
        others = torch.scatter(others, 0, pred_id_cat.unsqueeze(0).T, pred_label)
        others = (1 - alpha) * (others + self_label
                                ) * self_norm + alpha * self_label0
        others = others / torch.sum(others, -1, keepdim=True)
        nxt_y[batch] = others.numpy()

    # prepare labels
    N = graph.num_nodes()
    C = 153
    y = np.zeros((N, C), dtype="float32")
    y0 = np.zeros((N, C), dtype="float32")
    nxt_y = np.zeros((N, C), dtype="float32")

    train_idx = split['train'].tolist()
    val_idx = split["valid"].tolist()
    test_idx = split["test"].tolist()
    y0[val_idx] = valid_label
    y0[test_idx] = test_label

    y[val_idx] = valid_label
    y[test_idx] = test_label

    # remask mplp 只用了标签的paper，编号系统不一致，这里做一个对应
    # num_paperwithlabel = len(train_idx) + len(val_idx) + len(test_idx)
    # mask_paperwithlabel = np.full_like(labels, -1)
    # mask_paperwithlabel[np.concatenate([train_idx, val_idx, test_idx])] = np.arange(num_paperwithlabel)

    # remask = np.zeros(num_paperwithlabel, dtype=np.int32) - 1
    # remask[np.arange(num_paperwithlabel)] = np.concatenate([train_idx, val_idx, test_idx]

    remask = np.concatenate([train_idx, val_idx, test_idx])


    # edges of graph]

    # for i in range(5): #把该fold之外其他4个加到训练,该fold测试
    #     if i == fold_id:
    #         continue
    #     train_idx.extend(
    #         np.load(os.path.join(seed_path, "cv-%s" % i, "idx_valid.npy")).tolist())
    # train_idx = np.array(train_idx, dtype="int32")
    train_idx = remask[np.load(os.path.join(fpath, "idx_train.npy"))]

    val_idx = remask[np.load(os.path.join(fpath, "idx_valid.npy"))]
    test_idx = split['test']

    # set gold label

    y[train_idx].fill(0)
    y[train_idx, labels[train_idx].astype("int32")] = 1 #train_idx的标签转成one-hot

    y0[train_idx].fill(0)
    y0[train_idx, labels[train_idx].astype("int32")] = 1

    def smooth(y0, y, nxt_y, alpha=0.2):
        nodes = train_idx.tolist() + val_idx.tolist() + test_idx.tolist()
        pbar = tqdm(total=len(nodes))
        batch_size = 50000
        batch_no = 0
        nxt_y.fill(0)

        while batch_no < len(nodes):
            batch = nodes[batch_no:batch_no + batch_size]
            batch = torch.from_numpy(np.array(batch, dtype="int64"))
            # batch = paddle.to_tensor(batch, dtype="int64")
            aggr(batch, y, nxt_y, y0, alpha)
            batch_no += batch_size
            pbar.update(batch_size)

    

    best_acc = 0
    hop = 0

    train_label = labels[train_idx]
    train_pred = y[train_idx]
    train_pred = np.argmax(train_pred, -1)
    train_acc = evaluator.eval({
        'y_true': train_label,
        'y_pred': train_pred
    })['acc']
    print("Hop", hop, "alpha", alpha, "Train Acc", train_acc)

    valid_label = labels[val_idx]
    valid_pred = y[val_idx]
    valid_pred = np.argmax(valid_pred, -1)
    valid_acc = evaluator.eval({
        'y_true': valid_label,
        'y_pred': valid_pred
    })['acc']
    print("Hop", hop, "alpha", alpha, "Valid Acc", valid_acc)

    if valid_acc > best_acc:
        np.save(os.path.join(fpath, "y_pred_valid.npy"), y[val_idx])
        np.save(os.path.join(fpath, "y_pred_test.npy"), y[test_idx])
        # np.save(os.path.join(fpath, "idx_valid.npy"), val_idx)
        best_acc = valid_acc

    for hop in range(1, 5):
        smooth(y0, y, nxt_y, alpha)
        nxt_y, y = y, nxt_y

        y[train_idx].fill(0)
        y[train_idx, labels[train_idx].astype("int32")] = 1

        train_label = labels[train_idx]
        train_pred = y[train_idx]
        train_pred = np.argmax(train_pred, -1)
        train_acc = evaluator.eval({
            'y_true': train_label,
            'y_pred': train_pred
        })['acc']
        print("Hop", hop, "alpha", alpha, "Train Acc", train_acc)

        valid_label = labels[val_idx]
        valid_pred = y[val_idx]
        valid_pred = np.argmax(valid_pred, -1)
        valid_acc = evaluator.eval({
            'y_true': valid_label,
            'y_pred': valid_pred
        })['acc']
        print("Hop", hop, "alpha", alpha, "Valid Acc", valid_acc)

        if valid_acc > best_acc:
            np.save(os.path.join(fpath, "y_pred_valid.npy"), y[val_idx])
            np.save(os.path.join(fpath, "y_pred_test.npy"), y[test_idx])
            # np.save(os.path.join(fpath, "idx_valid.npy"), val_idx)
            best_acc = valid_acc

        # if valid_acc > best_acc:
        #     np.save("result/" + save_model_name + "/val_%s_pred.npy" % (fold_id),
        #             y[val_idx])
        #     np.save("result/" + save_model_name + "/test_%s.npy" % (fold_id),
        #             y[test_idx])
        #     np.save("result/" + save_model_name + "/valid_%s.npy" % (fold_id),
        #             val_idx)
        #     best_acc = valid_acc
