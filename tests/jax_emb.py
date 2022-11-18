import numpy as np
import glob
import dill
import os
from ogb.lsc import MAG240MDataset, MAG240MEvaluator

if __name__=="__main__":
    # 验证验证集的效果 
    root = 'dataset_path'
    # jax_feat_train_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/jax_em/one_it_train_set"
    # jax_feat_valid_test_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/jax_em/one_it_valid_test_set"
    jax_valid_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/jax_em/valid_only"
    dataset = MAG240MDataset(root)
    split_nids = dataset.get_idx_split()
    node_ids = np.concatenate([split_nids['train'], split_nids['valid'], split_nids['test-whole']])
    train_ids, valid_ids, test_ids = split_nids['train'], split_nids['valid'], split_nids['test-whole']

    y_pred = []
    idxs = []
    for fpath in glob.glob(os.path.join(jax_valid_path, 'valid*.dill')):
        print(fpath)
        inf = dill.load(open(fpath, 'rb'))
        idxs.append(inf[0])
        y_pred.append(inf[3])
    idxs = np.concatenate(idxs, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_pred_sort = y_pred[np.argsort(idxs)]

    # labels = dataset.paper_label
    # y_true = labels[valid_ids]


    # evaluator = MAG240MEvaluator()
    # acc = evaluator.eval(
    #     {'y_true': y_true, 'y_pred': y_pred_sort.argmax(axis=1)}
    # )['acc']
    # print("valid accurate: %.4f" % acc)
    

    # 替换掉7966结果中的valid向量表示
    jax_emb_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/jax_em/x_jax_153_yuan.npy" #这个里面测试集在生成表示时有知识泄露的情况
    jax_emb = np.load(jax_emb_path)
    jax_emb[len(train_ids):len(train_ids)+len(valid_ids)] = y_pred_sort
    np.save("/data/zhaohuanjing/mag/dataset_path/mplp_data/jax_em/emb/x_jax_153_new.npy", jax_emb) #这里是正确的验证集结果