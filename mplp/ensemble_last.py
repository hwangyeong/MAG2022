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
# output_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/mplp/outputs/subm"
output_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/mplp/outputs/try/tests"
# graph_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/preprocess/ppgraph_cite.bin"
# graph_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/preprocess/paper_coauthor_paper_symmetric_jc0.5_pgl.bin"
# y_re_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/mplp/data/y_base.npy"
jax_feat_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/feature/x_jax_153.npy"
# method_weight = [
#     ("rgat1024_label", 0.2),
#     # "rgat1024_label_seed1",
#     ("rgat1024_label_m2v_feat_fhid", 1),
#     ("rgat1024_label_m2v_feat_fhid_seed1", 0.8),
#     ("rgat1024_label_m2v_feat_fhid_seed2", 0.8),
#     ("rgat1024_label_m2v_feat_fhid_seed3", 1),
#     ("rgat1024_label_m2v_feat_fhid_seed4", 1),
#     ("rgat1024_label_m2v_feat_fhid_seed5", 1),

#     ("rgat1024_label_m2v_feat_3rgat1024", 0.9),
#     ("rgat1024_label_m2v_feat_3rgat1024_fhid", 1),
#     ("rgat1024_label_m2v_feat_3rgat1024_fhid_seed1", 0.2),
#     ("rgat1024_label_m2v_feat_3rgat1024_fhid_seed2", 1),

#     ("rgat1024_label_m2v_feat_fhid_hid1024", 0.8),
#     ("label_m2v_3rgat1024", 0.6)
# ]

method_weight = [
    ("rgat1024_label", 1),
    # "rgat1024_label_seed1",
    ("rgat1024_label_m2v_feat_fhid", 1),
    ("rgat1024_label_m2v_feat_fhid_seed1", 1),
    ("rgat1024_label_m2v_feat_fhid_seed2", 1),
    ("rgat1024_label_m2v_feat_fhid_seed3", 1),
    ("rgat1024_label_m2v_feat_fhid_seed4", 1),
    ("rgat1024_label_m2v_feat_fhid_seed5", 1),

    ("rgat1024_label_m2v_feat_3rgat1024", 1),
    ("rgat1024_label_m2v_feat_3rgat1024_fhid", 1),
    ("rgat1024_label_m2v_feat_3rgat1024_fhid_seed1", 1),
    ("rgat1024_label_m2v_feat_3rgat1024_fhid_seed2", 1),

    ("rgat1024_label_m2v_feat_fhid_hid1024", 1),
    ("label_m2v_3rgat1024", 1),
    ("rgat1024_label_m2v_feat_3rgat1024_jax_fhid", 1),
    # ("rgat1024_label_m2v_feat_jax_fhid", 1),
]

# [0.74306379 0.69978153 0.13868557 0.25365812 0.29762248 0.6796287
#  0.63549923 0.94590502 0.78731705 0.3785609  0.88095213 0.09700508
#  0.96977807]
# [0.10251475 0.37978563 0.07466749 0.01261141 0.30945725 0.83856793
#  0.73538991 0.20788492 0.66306799 0.08586953 0.94942823 0.26100088
#  0.72331826]
# [0.38223486 0.94927511 0.58175651 0.08372206 0.8688711  0.62926486
#  0.50473214 0.34752083 0.90149966 0.24608079 0.64669699 0.43034616
#  0.96366635]
# [ 0.36515168  0.9994153   0.50033305 -0.01006799  0.7847122   0.53560083
#   0.46611014  0.37300038  0.94927571  0.31458275  0.57553175  0.51911372
#   0.95768499]


# [0.9962423  0.25316235 0.49807109 0.73533163 0.74610756 0.12260966
#  0.31459612 0.16007702 0.13713396 0.39447655 0.10983136 0.3697219
#  0.05206976 0.87245373]

# rdw = [ 0.5571511,   0.16183978,  0.39309341,  0.13837567,  0.30907162, -0.05777938,
#  -0.15232993,  0.47739837,  0.38919884,  0.43782636,  0.16724044,  0.25951808,
#   0.82565718,  1.01226051]
# rdw = [1]* 14
# [-0.62325543  0.55098525  0.71992763  0.33087125 -0.78525913 -0.48330705
#  -0.38835018  0.28048456  0.35506171 -0.76298339 -0.43859547  0.78012929
#   0.52321169  0.99038397]
# [0.8448389755811809, 0.4292764741877291, -0.07758990434454449, -0.6202926642334756, 0.5359053570007746, 0.6750408297350976, -0.4247260917094249, 0.9113951040188237, -0.9321564210819644, -0.5166313166259422, 0.41108506002713185, -0.2000819972069492, -0.4311026361036854, 0.04120428628793782, 0.8834928157101933]

#最后的权重
# rdw = [1] * 15
rdw = [-0.4877865744749357, -0.346649996233688, -0.37787874613394634, -0.5604140184991806, 0.8799434501215753, 0.46518827818407305, -0.3769751431472341, 0.8269823445831963, -0.1954645260389538, 0.1888043427524997, 0.6469049538984475, -0.2883804174068405, -0.25362940422525115, 0.48079288659829533, 0.8627485259401018]
dataset = MAG240MDataset(root=dataset_path)
split_nids = dataset.get_idx_split()
node_ids = np.concatenate([split_nids['train'], split_nids['valid']])

y_true_train = dataset.paper_label[dataset.get_idx_split('train')]
y_true_valid = dataset.paper_label[dataset.get_idx_split('valid')]
y_all = np.concatenate([y_true_train, y_true_valid])
evaluator = MAG240MEvaluator()

jax_feat = np.load(jax_feat_path)
y_pred_val_jax = jax_feat[len(y_true_train):len(y_true_train) + len(y_true_valid)]
y_pred_test_jax = jax_feat[len(y_true_train) + len(y_true_valid):]

# while True:
    # rdw = [np.random.uniform(-1.,1.) for ii in range(16)]
# rdw = [-0.4877865744749357, -0.346649996233688, -0.37787874613394634, -0.5604140184991806, 0.8799434501215753, 0.46518827818407305, -0.3769751431472341, 0.8269823445831963, -0.1954645260389538, 0.1888043427524997, 0.6469049538984475, -0.2883804174068405, -0.25362940422525115, 0.48079288659829533, 0.48079288659829533, 0.8627485259401018]
# rdw = [-0.64018426, 
#        -0.46992505, 
#        -0.35731564, 
#        -0.58023838,  
#        0.88282795,  
#        0.58422976,
#        -0.50803445,  
#        0.96858588, 
#        -0.22182822,  
#        0.18403387,  
#        0.68290918, 
#        -0.11743891,
#        -0.30712759,  
#        0.81955227,  
#        0.216809,    
#        1.43663889
# ]

    # rde = [np.random.uniform(-0.2,0.2) for ii in range(len(rdw))]
    # rdw = np.array(rdw) + rde

method_weight = [(x[0],rdw[ii]) for ii,x in enumerate(method_weight)]

print("\nEvaluating at validation dataset...")
y_pred_valid_all = []
acc_all = []
for ii, (method, weight) in enumerate(method_weight):
    method_path = os.path.join(input_path, method)
    y_pred_v = []
    idx_v = []
    # print("\n %d %s..." % (ii,method))
    for fpath in glob.glob(os.path.join(method_path, 'cv-*')):
        # print("Loading predictions from %s" % fpath)
        y = torch.as_tensor(np.load(os.path.join(fpath, "y_pred_valid.npy"))).softmax(axis=1).numpy()
        y_pred_v.append(y)
        idx = np.load(os.path.join(fpath, "idx_valid.npy"))
        idx_v.append(idx)
    idx_v = np.concatenate(idx_v, axis=0)
    y_pred_v = np.concatenate(y_pred_v, axis=0)
    y_true_v = y_all[idx_v]
    y_pred_valid = y_pred_v[np.argsort(idx_v)]
    y_pred_valid_all.append(y_pred_valid * weight)
    np.save(os.path.join(method_path, 'y_pred_valid_all.npy'), y_pred_valid)

    acc = evaluator.eval(
        {'y_true': y_true_v, 'y_pred': y_pred_v.argmax(axis=1)}
    )['acc']
    acc_all.append(acc)
    # print("valid accurate: %.4f" % acc)

# add jax feats
weight_jax = rdw[-1]
y_pred_valid_all.append(y_pred_val_jax * weight_jax)

# print("\nvalid accurate distribution: %.4f +/- %.4f" % (np.mean(acc_all), np.std(acc_all)))

nsample, ndim = y_pred_valid_all[0].shape
# y_pred_valid_all = np.concatenate(y_pred_valid_all).reshape((-1, nsample, ndim)).mean(axis=0)
y_pred_valid_all = np.concatenate(y_pred_valid_all).reshape((-1, nsample, ndim)).sum(axis=0)
acc = evaluator.eval(
    {'y_true': y_true_valid, 'y_pred': y_pred_valid_all.argmax(axis=1)}
)['acc']
print("valid ensemble (average) accurate: %.4f" % acc)

    # if acc > 0.7955:
    #     print(rdw)
    #     break
    

y_pred_test_all = []
for ii, (method, weight) in enumerate(method_weight):
    method_path = os.path.join(input_path, method)
    y_pred = []
    for fpath in glob.glob(os.path.join(method_path, 'cv-*')):
        y = torch.as_tensor(np.load(os.path.join(fpath, "y_pred_test.npy"))).softmax(axis=1).numpy()
        y_pred.append(y)
    nsample, ndim = y_pred[0].shape
    y_pred = np.concatenate(y_pred).reshape((-1, nsample, ndim)).mean(axis=0)
    y_pred_test_all.append(y_pred * weight)

y_pred_test_all.append(y_pred_test_jax * weight_jax)

y_pred_test = np.concatenate(y_pred_test_all).reshape((-1, nsample, ndim)).sum(axis=0)
res = {'y_pred': y_pred_test.argmax(axis=1)}

# process test-challenge
test_idx = split_nids['test-whole']
# print(test_idx.shape)
test_challenge_idx = split_nids['test-challenge']
size = int(test_idx.max()) + 1
test_challenge_mask = torch.zeros(size, dtype=torch.bool)
test_challenge_mask[test_challenge_idx] = True
test_challenge_mask = test_challenge_mask[test_idx]

res_challenge = {}
res_challenge['y_pred'] = res['y_pred'][test_challenge_mask]

print("Saving SUBMISSION to %s" % output_path)
evaluator.save_test_submission(res_challenge, output_path, mode="test-challenge")

#process test-dev
res_dev = {}
test_dev_mask = ~test_challenge_mask
res_dev['y_pred'] = res['y_pred'][test_dev_mask]
print("Saving SUBMISSION to %s" % output_path)
evaluator.save_test_submission(res_dev, output_path, mode="test-dev")
