import sys
import os
import re
import glob
import numpy as np

from ogb.lsc import MAG240MDataset, MAG240MEvaluator
import torch


# dataset_path, input_path, output_path = sys.argv[1:4]

dataset_path = "/data/zhaohuanjing/mag/dataset_path"

input_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/mplp/outputs"
# graph_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/preprocess/ppgraph_cite.bin"
# graph_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/preprocess/paper_coauthor_paper_symmetric_jc0.5_pgl.bin"
# y_re_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/mplp/data/y_base.npy"

method_name = [
    "rgat1024_label",
    # "rgat1024_label_seed1",
    "rgat1024_label_m2v_feat_fhid",
    "rgat1024_label_m2v_feat_fhid_seed1",
    "rgat1024_label_m2v_feat_fhid_seed2",
    "rgat1024_label_m2v_feat_fhid_seed3",
    "rgat1024_label_m2v_feat_fhid_seed4",
    "rgat1024_label_m2v_feat_fhid_seed5",

    "rgat1024_label_m2v_feat_3rgat1024",
    "rgat1024_label_m2v_feat_3rgat1024_fhid",
    # "rgat1024_label_m2v_feat_fhid_hid1024",
]

dataset = MAG240MDataset(root=dataset_path)
split_nids = dataset.get_idx_split()
node_ids = np.concatenate([split_nids['train'], split_nids['valid']])

y_true_train = dataset.paper_label[dataset.get_idx_split('train')]
y_true_valid = dataset.paper_label[dataset.get_idx_split('valid')]
y_all = np.concatenate([y_true_train, y_true_valid])

evaluator = MAG240MEvaluator()

print("\nEvaluating at validation dataset...")
y_pred_valid_all = []
acc_all = []
for method in method_name:
    method_path = os.path.join(input_path, method)
    y_pred_v = []
    idx_v = []
    print("\n %s..." % method)
    for fpath in glob.glob(os.path.join(method_path, 'cv-*')):
        print("Loading predictions from %s" % fpath)
        y = torch.as_tensor(np.load(os.path.join(fpath, "y_pred_valid.npy"))).softmax(axis=1).numpy()
        y_pred_v.append(y)
        idx = np.load(os.path.join(fpath, "idx_valid.npy"))
        idx_v.append(idx)
    idx_v = np.concatenate(idx_v, axis=0)
    y_pred_v = np.concatenate(y_pred_v, axis=0)
    y_true_v = y_all[idx_v]
    y_pred_valid = y_pred_v[np.argsort(idx_v)]
    y_pred_valid_all.append(y_pred_valid)
    np.save(os.path.join(method_path, 'y_pred_valid_all.npy'), y_pred_valid)

    acc = evaluator.eval(
        {'y_true': y_true_v, 'y_pred': y_pred_v.argmax(axis=1)}
    )['acc']
    acc_all.append(acc)
    print("valid accurate: %.4f" % acc)
print("valid accurate distribution: %.4f +/- %.4f" % (np.mean(acc_all), np.std(acc_all)))

nsample, ndim = y_pred_valid_all[0].shape
y_pred_valid_all = np.concatenate(y_pred_valid_all).reshape((-1, nsample, ndim)).mean(axis=0)
acc = evaluator.eval(
    {'y_true': y_true_valid, 'y_pred': y_pred_valid_all.argmax(axis=1)}
)['acc']
print("valid ensemble (average) accurate: %.4f" % acc)
    

# # for seed_path in glob.glob(os.path.join(input_path, 'seed*')):
# seed_path = input_path
# y_pred_v = []
# idx_v = []
# for fpath in glob.glob(os.path.join(seed_path, 'cv-*')):
#     print("Loading predictions from %s" % fpath)
#     y = torch.as_tensor(np.load(os.path.join(fpath, "y_pred_valid.npy"))).softmax(axis=1).numpy()
#     y_pred_v.append(y)
#     idx = np.load(os.path.join(fpath, "idx_valid.npy"))
#     idx_v.append(idx)
# idx_v = np.concatenate(idx_v, axis=0)
# y_pred_v = np.concatenate(y_pred_v, axis=0)
# y_true_v = y_all[idx_v]
# # y_pred_valid_all.append(y_pred_v[np.argsort(idx_v)])
# y_pred_valid_all = y_pred_v[np.argsort(idx_v)]

# acc = evaluator.eval(
#     {'y_true': y_true_v, 'y_pred': y_pred_v.argmax(axis=1)}
# )['acc']
# acc_all.append(acc)
# print("valid accurate: %.4f" % acc)

# np.save(os.path.join(seed_path, 'y_pred_valid_all'), y_pred_valid_all)


print("\nEnsembling predictions for test dataset...")
filenames = glob.glob(os.path.join(input_path, "cv-*", "y_pred_test.npy"))

y_pred_test_all = []
for fname in filenames:
    print(fname)
    y = torch.as_tensor(np.load(fname)).softmax(axis=1).numpy()
    y_pred_test_all.append(y)
nsample, ndim = y_pred_test_all[0].shape

y_pred_test = np.concatenate(y_pred_test_all).reshape((-1, nsample, ndim)).mean(axis=0)
res = {'y_pred': y_pred_test.argmax(axis=1)}

# process test-challenge
test_idx = split_nids['test-whole']
# print(test_idx.shape)
test_challenge_idx = split_nids['test-challenge']
size = int(test_idx.max()) + 1
test_challenge_mask = torch.zeros(size, dtype=torch.bool)
test_challenge_mask[test_challenge_idx] = True
test_challenge_mask = test_challenge_mask[test_idx]
res['y_pred'] = res['y_pred'][test_challenge_mask]

print("Saving SUBMISSION to %s" % output_path)
evaluator.save_test_submission(res, output_path, mode="test-challenge")
