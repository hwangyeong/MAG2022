#!/usr/bin/env python3
import argparse
from pyexpat import features
import sys
import os
import errno

import random
import tqdm
from collections import defaultdict

from ogb.lsc import MAG240MDataset
import numpy as np
import dgl
import torch
import logging


logging.basicConfig(
     format= '%(asctime)s %(levelname)s %(module)s : %(message)s',
     level=logging.INFO
)
logger = logging.getLogger(__name__)


def seed_everything(seed):
    """Set seed for all possible random place.
    """
    logger.info("Set ALL possible random seed to %d" % seed)
    random.seed(seed)
    np.random.seed(seed)
    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdirs(path):
    try:
        logger.info("Creating output path: %s" % path)
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST and os.path.isdir(path):
            raise e


def calc_randomwalk_label_features(graph, node_ids, metapath, labels, num_classes=153, num_walkers=160, batch_size=1024):
    res = []
    for i in tqdm.tqdm(range(0, len(node_ids), batch_size)):
        nids0 = np.array(node_ids[i:(i + batch_size)])
        nids = np.repeat(nids0, num_walkers)
        traces, _ = dgl.sampling.random_walk(graph, nids, metapath=metapath)

        traces = traces[traces[:, 0] != traces[:, -1]] #首节点尾节点相同的去掉？ 没懂
        m = defaultdict(list)
        for sid, *_, did in traces.numpy():
            if did >= 0:
                m[sid].append(did)
        mapper = {i: np.array(j) for i, j in m.items()}

        feat = np.zeros((len(nids0), num_classes), dtype=np.float32)
        for i, sid in enumerate(nids0):
            if sid in mapper.keys():
                dids = mapper[sid]
                lbs = labels[dids]
                mask = (dids != sid) & (lbs >= 0)
                if mask.sum() < 1.0e-6:
                    feat[i, :] = 1. / num_classes
                else:
                    ft = np.zeros((len(dids), num_classes), dtype=np.float32)
                    ft[mask, lbs[mask].astype(np.int64)] = 1
                    feat[i, :] = ft.sum(axis=0) / ft.sum()
            else:
                feat[i, :] = 1. / num_classes
        res.append(feat)
    return np.concatenate(res, axis=0)


def calc_randomwalk_feat_features(graph, node_ids, metapath, features, feature_dim=768, num_walkers=160, batch_size=1024):
    res = []
    for i in tqdm.tqdm(range(0, len(node_ids), batch_size)):
        nids0 = np.array(node_ids[i:(i + batch_size)])
        nids = np.repeat(nids0, num_walkers)
        traces, _ = dgl.sampling.random_walk(graph, nids, metapath=metapath)

        traces = traces[traces[:, 0] != traces[:, -1]]
        m = defaultdict(list)
        for sid, *_, did in traces.numpy():
            if did >= 0:
                m[sid].append(did)
        mapper = {i: np.array(j) for i, j in m.items()}

        feat = np.zeros((len(nids0), feature_dim), dtype=np.float32)
        for i, sid in enumerate(nids0):
            if sid in mapper.keys():
                feat[i, :] = features[mapper[sid]].astype(np.float32).mean(axis=0)
            else:
                feat[i, :] = 0
        res.append(feat)
    return np.concatenate(res, axis=0)


def calc_randomwalk_topk_label_features(graph, node_ids, metapath, labels, num_classes=153, num_walkers=160, topk=10, batch_size=1024):
    res = []
    for i in tqdm.tqdm(range(0, len(node_ids), batch_size)):
        nids0 = np.array(node_ids[i:(i + batch_size)])
        nids = np.repeat(nids0, num_walkers)
        traces, _ = dgl.sampling.random_walk(graph, nids, metapath=metapath)

        traces = traces[traces[:, 0] != traces[:, -1]]
        m = defaultdict(list)
        for sid, *_, did in traces.numpy():
            if did >= 0:
                m[sid].append(did)
        mapper = {i: np.array(j) for i, j in m.items()}

        feat = np.zeros((len(nids0), num_classes), dtype=np.float32)
        for i, sid in enumerate(nids0):
            if sid in mapper.keys():
                dids = mapper[sid]
                dids, cnts = np.unique(dids, return_counts=True)
                itk = np.argsort(cnts)[-topk:]
                dids, cnts = dids[itk], cnts[itk]
                lbs = labels[dids]
                mask = (dids != sid) & (lbs >= 0)
                if mask.sum() < 1.0e-6:
                    feat[i, :] = 1. / num_classes
                else:
                    ft = np.zeros((len(dids), num_classes), dtype=np.float32)
                    ft[mask, lbs[mask].astype(np.int64)] = 1
                    ft *= cnts.reshape((-1, 1))
                    feat[i, :] = ft.sum(axis=0) / cnts.sum()
            else:
                feat[i, :] = 1. / num_classes
        res.append(feat)
    return np.concatenate(res, axis=0)


def calc_randomwalk_topk_feat_features(graph, node_ids, metapath, features, feature_dim=768, num_walkers=160, topk=10, batch_size=1024):
    res = []
    for i in tqdm.tqdm(range(0, len(node_ids), batch_size)):
        nids0 = np.array(node_ids[i:(i + batch_size)])
        nids = np.repeat(nids0, num_walkers)
        traces, _ = dgl.sampling.random_walk(graph, nids, metapath=metapath)

        traces = traces[traces[:, 0] != traces[:, -1]]
        m = defaultdict(list)
        for sid, *_, did in traces.numpy():
            if did >= 0:
                m[sid].append(did)
        mapper = {i: np.array(j) for i, j in m.items()}

        feat = np.zeros((len(nids0), feature_dim), dtype=np.float32)
        for i, sid in enumerate(nids0):
            if sid in mapper.keys():
                dids = mapper[sid]
                dids, cnts = np.unique(dids, return_counts=True)
                itk = np.argsort(cnts)[-topk:]
                dids, cnts = dids[itk], cnts[itk]
                ft = features[dids].astype(np.float32) * cnts.reshape((-1, 1))
                feat[i, :] = ft.sum(axis=0) / cnts.sum()
            else:
                feat[i, :] = 0
        res.append(feat)
    return np.concatenate(res, axis=0)


def calc_neighborsample_label_features(graph, node_ids, metapath, labels, num_classes=153):
    feat = np.zeros((len(node_ids), num_classes), dtype=np.float32)
    node_ids = np.array(node_ids)
    for i, nids0 in enumerate(tqdm.tqdm(node_ids)):
        nids = nids0
        for mp in metapath:
            _, nids = map(lambda x: x.numpy(), graph.out_edges(nids, form='uv', etype=mp))
        nids = np.unique(nids[(nids != nids0)])

        if len(nids) > 0:
            lbs = labels[nids]
            lbs = lbs[(lbs >= 0)].astype(np.int64)
            if len(lbs) == 0:
                feat[i, :] = 1. / num_classes
            else:
                ft = np.zeros((len(lbs), num_classes), dtype=np.float32)
                ft[list(range(len(lbs))), lbs] = 1
                feat[i, :] = ft.sum(axis=0) / ft.sum()
        else:
            feat[i, :] = 1. / num_classes
    return feat


def calc_neighborsample_feat_features(graph, node_ids, metapath, features, feature_dim=768):
    feat = np.zeros((len(node_ids), feature_dim), dtype=np.float32)
    node_ids = np.array(node_ids)
    for i, nids0 in enumerate(tqdm.tqdm(node_ids)):
        nids = nids0
        for mp in metapath:
            _, nids = map(lambda x: x.numpy(), graph.out_edges(nids, form='uv', etype=mp))
        nids = np.unique(nids[(nids != nids0)])

        if len(nids) > 0:
            feat[i, :] = features[nids].astype(np.float32).mean(axis=0)
        else:
            feat[i, :] = 0
    return feat


def calc_neighborsample_heter_feat_features(graph, node_ids, metapath, feat_paper, target_type, 
                        feat_author=None, feat_institution=None, feature_dim=768):
    feat = np.zeros((len(node_ids), feature_dim), dtype=np.float32)
    if target_type == 'p':
        features = feat_paper
    elif target_type == 'a':
        features = feat_author
    elif target_type == 'i':
        features = feat_institution
    node_ids = np.array(node_ids)
    for i, nids0 in enumerate(tqdm.tqdm(node_ids)):
        nids = nids0
        for mp in metapath:
            _, nids = map(lambda x: x.numpy(), graph.out_edges(nids, form='uv', etype=mp))
        nids = np.unique(nids[(nids != nids0)])

        if len(nids) > 0:
            feat[i, :] = features[nids].astype(np.float32).mean(axis=0)
        else:
            feat[i, :] = 0
    return feat


def calc_neighborsample_filter_label_features(graph, node_ids, metapath, labels, num_classes=153, ftype='least', num_common=2):
    if ftype not in {'least', 'common'}:
        raise ValueError("Unknown ftype: %r, only support 'least' and 'common'" % ftype)
    if len(metapath) != 2:
        raise ValueError("metapath should with length 2: %r" % metapath)

    feat = np.zeros((len(node_ids), num_classes), dtype=np.float32)
    node_ids = np.array(node_ids)
    for i, nids0 in enumerate(tqdm.tqdm(node_ids)):
        dids = nids0
        for mp in metapath:
            sids, dids = map(lambda x: x.numpy(), graph.out_edges(dids, form='uv', etype=mp))

        if ftype == 'least':
            nids, cnts = np.unique(sids, return_counts=True)
            sid = nids[np.argmin(cnts)]  # least middle
            nids = dids[sids == sid]
            nids = nids[(nids != nids0)]
        else:
            nids, cnts = np.unique(dids, return_counts=True)
            nids = nids[cnts >= num_common]
            nids = nids[(nids != nids0)]

        if len(nids) > 0:
            lbs = labels[nids]
            lbs = lbs[(lbs >= 0)].astype(np.int64)
            if len(lbs) == 0:
                feat[i, :] = 1. / num_classes
            else:
                ft = np.zeros((len(lbs), num_classes), dtype=np.float32)
                ft[list(range(len(lbs))), lbs] = 1
                feat[i, :] = ft.sum(axis=0) / ft.sum()
        else:
            feat[i, :] = 1. / num_classes
    return feat


def calc_neighborsample_filter_feat_features(graph, node_ids, metapath, features, feature_dim=768, ftype='least', num_common=2):
    if ftype not in {'least', 'common'}:
        raise ValueError("Unknown ftype: %r, only support 'least' and 'common'" % ftype)
    if len(metapath) != 2:
        raise ValueError("metapath should with length 2: %r" % metapath)

    feat = np.zeros((len(node_ids), feature_dim), dtype=np.float32)
    node_ids = np.array(node_ids)
    for i, nids0 in enumerate(tqdm.tqdm(node_ids)):
        dids = nids0
        for mp in metapath:
            sids, dids = map(lambda x: x.numpy(), graph.out_edges(dids, form='uv', etype=mp))

        if ftype == 'least':
            nids, cnts = np.unique(sids, return_counts=True)
            sid = nids[np.argmin(cnts)]  # least middle
            nids = dids[sids == sid]
            nids = nids[(nids != nids0)]
        else:
            nids, cnts = np.unique(dids, return_counts=True)
            nids = nids[cnts >= num_common]
            nids = nids[(nids != nids0)]

        if len(nids) > 0:
            feat[i, :] = features[nids].astype(np.float32).mean(axis=0)
        else:
            feat[i, :] = 0
    return feat

def feature_project(features, embed_size):
    for ii in range(len(features)):
        if features[ii].size(1) != embed_size:
            rand_weight = torch.Tensor(features[ii].size(1), embed_size).uniform_(-0.5, 0.5)
            features[ii] = features[ii] @ rand_weight

def load_data(datapath, feat_info, device=None):
    logger.info("Loading data from %s" % datapath)

    x_all = []
    for i, (fn, _, _) in enumerate(feat_info):
        logger.info("Loading features for %d: %s ..." % (i, fn.upper()))
        feat = []
        fname = os.path.join(datapath, '%s.npy' % fn)
        logger.info("Loading %s" % fname)
        feat = torch.from_numpy(np.load(fname)).to(device, torch.float32)
        x_all.append(feat)

    logger.info("Loading labels ...")
    fname = os.path.join(datapath, 'y_base.npy')
    logger.info("Loading %s" % fname)
    y_all = torch.from_numpy(np.load(fname)).to(device, torch.long)

    return x_all, y_all


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Feature Engineering for OGB-MAG240M',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # parser.add_argument('dataset_path', help='The directory of dataset')
    # parser.add_argument('graph_filename', help='The filename of input heterogeneous graph (coo or csr format)')
    # parser.add_argument('output_path', help='The directory of output data')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    logger.info(args)
    args.dataset_path = "/data/zhaohuanjing/mag/dataset_path"
    # args.feature_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/feature/"
    args.graph_filename = "/data/zhaohuanjing/mag/dataset_path/mplp_data/preprocess/dgl_graph_full_heterogeneous_csr.bin"
    args.output_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/feature/"
    args.preprocess_path = "/data/zhaohuanjing/mag/dataset_path/mplp_data/preprocess"
    args.gpu = True
    args.emb_size = 256

    mkdirs(args.output_path)
    seed_everything(args.seed)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    dataset = MAG240MDataset(root=args.dataset_path)
    num_feats = dataset.num_paper_features # 原paper属性维度

    # feat_info = [
    #     ('x_rgat_1024', 1024, 128),
    #     ('x_base', 768, 128),
    #     ('x_pcbpcp_rw_lratio', 153, 32),
    #     ('x_pcbpcp_rw_top10_lratio', 153, 32),
    #     ('x_pcp_rw_lratio', 153, 32),
    #     ('x_pcpcbp_rw_lratio', 153, 32),
    #     ('x_pcpcbp_rw_top10_lratio', 153, 32),
    #     ('x_pcpcp_rw_lratio', 153, 32),
    #     ('x_pcpcp_rw_top10_lratio', 153, 32),
    #     ('x_pwbawp_rw_lratio', 153, 32),
    #     ('x_pwbawp_rw_top10_lratio', 153, 32),
    #     ('x_pwbawp_ns_l_lratio', 153, 32),
    #     ('x_pwbawp_ns_c2_lratio', 153, 32),
    #     ('x_pwbawp_ns_c4_lratio', 153, 32)
    # ]

    # logger.info("A Total of %d different type features" % len(feat_info))
    # logger.info(feat_info)

    # x_all, y_all = load_data(args.input_path, feat_info, device)
    # logger.info("<=2019: %d, >=2020: %d" % (y_all.shape[0], x_all[0].shape[0] - y_all.shape[0]))




    node_ids = np.concatenate(
        [dataset.get_idx_split(c).astype(np.int64) for c in ['train', 'valid', 'test']]
    ) # 有标签的那些papers
    paper_feat = dataset.paper_feat
    paper_label = dataset.all_paper_label
    paper_year = dataset.all_paper_year

    feature_dim, num_classes = paper_feat.shape[1], dataset.num_classes
    author_feat = np.memmap(os.path.join(args.preprocess_path, 'author.npy'), mode='r', dtype='float16', shape=(dataset.num_authors, feature_dim))
    ins_feat = np.memmap(os.path.join(args.preprocess_path, 'inst.npy'), mode='r', dtype='float16', shape=(dataset.num_institutions, feature_dim))

    logger.info("Paper feature dimension: %d, Paper class number: %d" % (feature_dim, num_classes))

    graph = dgl.load_graphs(args.graph_filename)[0][0]
    graph = graph.formats(['csr'])  # when use crc format, out_edges return incorrect result

    # # 0. base
    # logger.info('base')
    # x_base, y_base = paper_feat[node_ids], paper_label[node_ids] # 取出有标签的那些文章的属性和标签
    # y_base = y_base[y_base >= 0]  # get ride of test
    # np.save(os.path.join(args.output_path, 'x_base.npy'), x_base)
    # np.save(os.path.join(args.output_path, 'y_base.npy'), y_base)

    # # 1. random walk
    # metapaths = {
    #     'pcp': ['cites'],
    #     'pcbp': ['cited_by'],
    #     'pcpcbp': ['cites', 'cited_by'],
    #     'pcbpcp': ['cited_by', 'cites'],
    #     'pcpcp': ['cites', 'cites'],
    #     'pcbpcbp': ['cited_by', 'cited_by'],
    #     'pwbawp': ['writed_by', 'writes']
    # }
    # for n, mp in metapaths.items():
    #     logger.info(n, mp)
    #     x = calc_randomwalk_feat_features(graph, node_ids, mp, paper_feat, feature_dim,
    #                                       num_walkers=160)
    #     np.save(os.path.join(args.output_path, 'x_%s_rw_fmean.npy' % n), x)
    #     x = calc_randomwalk_label_features(graph, node_ids, mp, paper_label, num_classes,
    #                                        num_walkers=160) # labels的rw的邻居的labels求平均
    #     np.save(os.path.join(args.output_path, 'x_%s_rw_lratio.npy' % n), x)

    # # 2. random walk topk
    # metapaths = {
    #     'pcp': ['cites'],
    #     'pcbp': ['cited_by'],
    #     'pcpcbp': ['cites', 'cited_by'],
    #     'pcbpcp': ['cited_by', 'cites'],
    #     'pcpcp': ['cites', 'cites'],
    #     'pcbpcbp': ['cited_by', 'cited_by'],
    #     'pwbawp': ['writed_by', 'writes']
    # }
    # topk = 10
    # for n, mp in metapaths.items():
    #     logger.info(n, mp)
    #     x = calc_randomwalk_topk_feat_features(graph, node_ids, mp, paper_feat, feature_dim,
    #                                            num_walkers=160, topk=topk)
    #     np.save(os.path.join(args.output_path, 'x_%s_rw_top%d_fmean.npy' % (n, topk)), x)
    #     x = calc_randomwalk_topk_label_features(graph, node_ids, mp, paper_label, num_classes,
    #                                             num_walkers=160, topk=topk)
    #     np.save(os.path.join(args.output_path, 'x_%s_rw_top%d_lratio.npy' % (n, topk)), x)

    # neighbor sample
    # metapaths = {
    #     'pcp': ['cites'],
    #     'pcbp': ['cited_by'],
    #     'pcpcbp': ['cites', 'cited_by'],
    #     'pcbpcp': ['cited_by', 'cites'],
    #     'pcpcp': ['cites', 'cites'],
    #     'pcbpcbp': ['cited_by', 'cited_by'],
    #     'pwbawp': ['writed_by', 'writes']
    # }
    metapaths = {
        'pwba': ['writed_by'],
        'pwbaawi': ['writed_by', 'affiliated_with'],
        'pcpwba': ['cites', 'writed_by'],
        'pcbpwba': ['cited_by', 'writed_by'],
    }
    for n, mp in metapaths.items():
        logger.info(n, mp)
        x = calc_neighborsample_heter_feat_features(graph, node_ids, mp, paper_feat, n[-1], 
                        feat_author=author_feat, feat_institution=ins_feat, feature_dim=768)
        # x = calc_neighborsample_feat_features(graph, node_ids, mp, paper_feat, feature_dim)
        np.save(os.path.join(args.output_path, 'x_%s_ns_fmean.npy' % n), x)
        x = calc_neighborsample_label_features(graph, node_ids, mp, paper_label, num_classes)
        # np.save(os.path.join(args.output_path, 'x_%s_ns_lratio.npy' % n), x)

    # # neighbor sample by 'least' or 'common'
    # metapaths = {
    #     'pwbawp': ['writed_by', 'writes']
    # }
    # for n, mp in metapaths.items():
    #     logger.info(n, mp)
    #     x = calc_neighborsample_filter_feat_features(graph, node_ids, mp, paper_feat, feature_dim,
    #                                                  ftype='common', num_common=2)
    #     np.save(os.path.join(args.output_path, 'x_%s_ns_c2_fmean.npy' % n), x)
    #     x = calc_neighborsample_filter_label_features(graph, node_ids, mp, paper_label, num_classes,
    #                                                   ftype='common', num_common=2)
    #     np.save(os.path.join(args.output_path, 'x_%s_ns_c2_lratio.npy' % n), x)

    #     x = calc_neighborsample_filter_feat_features(graph, node_ids, mp, paper_feat, feature_dim,
    #                                                  ftype='common', num_common=4)
    #     np.save(os.path.join(args.output_path, 'x_%s_ns_c4_fmean.npy' % n), x)
    #     x = calc_neighborsample_filter_label_features(graph, node_ids, mp, paper_label, num_classes,
    #                                                   ftype='common', num_common=4)
    #     np.save(os.path.join(args.output_path, 'x_%s_ns_c4_lratio.npy' % n), x)

    #     x = calc_neighborsample_filter_feat_features(graph, node_ids, mp, paper_feat, feature_dim,
    #                                                  ftype='least')
    #     np.save(os.path.join(args.output_path, 'x_%s_ns_l_fmean.npy' % n), x)
    #     x = calc_neighborsample_filter_label_features(graph, node_ids, mp, paper_label, num_classes,
    #                                                   ftype='least')
    #     np.save(os.path.join(args.output_path, 'x_%s_ns_l_lratio.npy' % n), x)

    logger.info("DONE")
