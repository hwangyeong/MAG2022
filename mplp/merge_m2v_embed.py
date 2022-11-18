import os
import numpy as np
import torch
from tqdm import tqdm
import glob
import dill
from ogb.lsc import MAG240MDataset, MAG240MEvaluator

if __name__ == '__main__':
    # process m2v from pgl method

    root = 'dataset_path'
    # m2v_input_path = '/data/zhaohuanjing/mag/dataset_path/mplp_data/mag240m_m2v_feat_from_pgl'
    # m2v_output_path = '/data/zhaohuanjing/mag/dataset_path/mplp_data/feature/pgl_m2v.npy'
    # dataset = MAG240MDataset(root)
    # N = (dataset.num_papers + dataset.num_authors + dataset.num_institutions)
    # part_num = (N + 1) // 10
    # start_idx = 0
    # node_chunk_size = 100000
    # m2v_merge_feat = np.memmap(m2v_output_path, dtype=np.float16, mode='w+', shape=(N, 64))
    # files = os.listdir(m2v_input_path)
    # files = sorted(files)
    # for idx, start_idx in enumerate(range(0, N, part_num)):
    #     end_idx = min(N, start_idx + part_num)
    #     f = os.path.join(m2v_input_path, files[idx])
    #     m2v_feat_tmp = np.memmap(f, dtype=np.float16, mode='r', shape=(end_idx - start_idx, 64))
    #     for i in tqdm(range(start_idx, end_idx, node_chunk_size)):
    #         j = min(i + node_chunk_size, end_idx)
    #         m2v_merge_feat[i: j] = m2v_feat_tmp[i - start_idx: j - start_idx]
    #     m2v_merge_feat.flush()
    #     del m2v_feat_tmp

    # m2v_feat_tmp = np.memmap(m2v_output_path, dtype=np.float16, mode='r', shape=(N, 64))
    # # 取一部分出来
    # split_nids = dataset.get_idx_split()
    # node_ids = np.concatenate([split_nids['train'], split_nids['valid'], split_nids['test-whole']])
    # m2v_feats_formplp = m2v_merge_feat[node_ids]

    # fpath = "/data/zhaohuanjing/mag/dataset_path/mplp_data/feature/x_m2v_64.npy"
    # np.save(fpath, m2v_feats_formplp)





    # # process pca feat from deepmind method
    # # load pca featurs
    # pca_feats = np.load(
    #         "/data/zhaohuanjing/mag/dataset_path/jax_process/data/preprocessed/merged_feat_from_paper_feat_pca_129.npy", mmap_mode="r")
    # dataset = MAG240MDataset(root)
    # split_nids = dataset.get_idx_split()
    # node_ids = np.concatenate([split_nids['train'], split_nids['valid'], split_nids['test-whole']])
    # pca_feats_formplp = pca_feats[node_ids]

    # fpath = "/data/zhaohuanjing/mag/dataset_path/mplp_data/feature/x_pca_129.npy"
    # np.save(fpath, pca_feats_formplp)



    # process jax feat from deepmind method
    jax_feat_train_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/jax_em/one_it_train_set"
    jax_feat_valid_test_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/jax_em/one_it_valid_test_set"
    dataset = MAG240MDataset(root)
    split_nids = dataset.get_idx_split()
    node_ids = np.concatenate([split_nids['train'], split_nids['valid'], split_nids['test-whole']])
    train_ids, valid_ids, test_ids = split_nids['train'], split_nids['valid'], split_nids['test-whole']

    y_pred = []
    idxs = []
    for fpath in glob.glob(os.path.join(jax_feat_train_path, 'valid*.dill')):
            print(fpath)
            inf = dill.load(open(fpath, 'rb'))
            idxs.append(inf[0])
            y_pred.append(inf[3])
    # idxs = np.concatenate(idxs, axis=0)
    # y_pred = np.concatenate(y_pred, axis=0)
    # y_pred_sort = y_pred[np.argsort(idxs)]

    for fpath in glob.glob(os.path.join(jax_feat_valid_test_path, 'valid*.dill')):
            print(fpath)
            inf = dill.load(open(fpath, 'rb'))
            idxs.append(inf[0])
            y_pred.append(inf[3])
    idxs = np.concatenate(idxs, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    # y_pred_sort = y_pred[np.argsort(idxs)]

    mask_train = np.zeros(len(node_ids), dtype=np.bool)
    mask_valid = np.zeros(len(node_ids), dtype=np.bool)
    mask_test = np.zeros(len(node_ids), dtype=np.bool)
    for ii in range(len(idxs)):
            if ii % 10000 == 0:
                    print(ii)
            node = idxs[ii] 
            if node in train_ids:
                    mask_train[ii] = True
            elif node in valid_ids:
                    mask_valid[ii] = True
            elif node in test_ids:
                    mask_test[ii] = True
            else:
                    print(node)
    
    train_idx = idxs[mask_train]
    valid_idx = idxs[mask_valid]
    test_idx = idxs[mask_test]

    y_pred_train = y_pred[mask_train][np.argsort(train_idx)]
    y_pred_valid = y_pred[mask_valid][np.argsort(valid_idx)]
    y_pred_test = y_pred[mask_test][np.argsort(test_idx)]

    y_pred = np.concatenate((y_pred_train, y_pred_valid, y_pred_test), axis=0)
    print(y_pred.shape)

    # fpath = "/data/zhaohuanjing/mag/dataset_path/mplp_data/feature/x_jax_153.npy"
    # np.save(fpath, y_pred)

    # print(len(split_nids['train']))
    # print(len(split_nids['valid']))
    # print(len(split_nids['test-whole']))
    # print(np.intersect1d(idxs, split_nids['train']).shape)
    # print(np.intersect1d(idxs, split_nids['valid']).shape)
    # print(np.intersect1d(idxs, split_nids['test-whole']).shape)
    # print(y_pred.shape)

    #save_test_challenge
    # # process test-challenge
    test_idx = split_nids['test-whole']
    # print(test_idx.shape)
    test_challenge_idx = split_nids['test-challenge']
    size = int(test_idx.max()) + 1
    test_challenge_mask = torch.zeros(size, dtype=torch.bool)
    test_challenge_mask[test_challenge_idx] = True
    test_challenge_mask = test_challenge_mask[test_idx]

    # res = {'y_pred': y_pred_test.argmax(axis=1)}
    # res['y_pred'] = res['y_pred'][test_challenge_mask]
    # output_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/mplp/outputs/try"
    # print("Saving SUBMISSION to %s" % output_path)
    # evaluator = MAG240MEvaluator()
    # evaluator.save_test_submission(res, output_path, mode="test-challenge")

    #process test-dev
    test_dev_mask = ~test_challenge_mask
    res = {'y_pred': y_pred_test.argmax(axis=1)}
    res['y_pred'] = res['y_pred'][test_dev_mask]
    output_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/mplp/outputs/try"
    print("Saving SUBMISSION to %s" % output_path)
    evaluator = MAG240MEvaluator()
    evaluator.save_test_submission(res, output_path, mode="test-dev")


