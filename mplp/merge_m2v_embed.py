import os
import numpy as np
from tqdm import tqdm
from ogb.lsc import MAG240MDataset

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





    # process pca feat from deepmind method
    # load pca featurs
    pca_feats = np.load(
            "/data/zhaohuanjing/mag/dataset_path/jax_process/data/preprocessed/merged_feat_from_paper_feat_pca_129.npy", mmap_mode="r")
    dataset = MAG240MDataset(root)
    split_nids = dataset.get_idx_split()
    node_ids = np.concatenate([split_nids['train'], split_nids['valid'], split_nids['test-whole']])
    pca_feats_formplp = pca_feats[node_ids]

    fpath = "/data/zhaohuanjing/mag/dataset_path/mplp_data/feature/x_pca_129.npy"
    np.save(fpath, pca_feats_formplp)



    # process mplp feat from deepmind method


