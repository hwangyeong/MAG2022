#!/usr/bin/env bash
export DGLBACKEND=pytorch
export CUDA_VISIBLE_DEVICES=7

MAG_BASE_PATH=/data/zhaohuanjing/mag/ # modify to your workspace

MAG_INPUT_PATH=$MAG_BASE_PATH/dataset_path/  # The MAG240M-LSC dataset should be placed here

MAG_CODE_PATH=$MAG_BASE_PATH/mplp/
# MAG_PREP_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/preprocess/
# MAG_RGAT_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/rgat/
# MAG_FEAT_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/feature/
MAG_MPLP_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/mplp/
# MAG_SUBM_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/submission/


# mkdir -p $MAG_PREP_PATH
# python $MAG_CODE_PATH/preprocess.py \
#         --rootdir $MAG_INPUT_PATH \
#         --author-output-path $MAG_PREP_PATH/author.npy \
#         --inst-output-path $MAG_PREP_PATH/inst.npy \
#         --graph-output-path $MAG_PREP_PATH \
#         --graph-as-homogeneous \
#         --full-output-path $MAG_PREP_PATH/full_feat.npy


# mkdir -p $MAG_FEAT_PATH
# python3 $MAG_CODE_PATH/feature.py \
#         $MAG_INPUT_PATH \
#         $MAG_PREP_PATH/dgl_graph_full_heterogeneous_csr.bin \
#         $MAG_FEAT_PATH \
#         --seed=42


# mkdir -p $MAG_RGAT_PATH
# python $MAG_CODE_PATH/rgat.py \
#         --rootdir $MAG_INPUT_PATH \
#         --graph-path $MAG_PREP_PATH/dgl_graph_full_homogeneous_csc.bin \
#         --full-feature-path $MAG_PREP_PATH/full_feat.npy \
#         --output-path $MAG_RGAT_PATH/ \
#         --epochs=100 \
#         --model-path $MAG_RGAT_PATH/model.pt \
#         # --submission-path $MAG_RGAT_PATH/


# mkdir -p $MAG_MPLP_PATH/data
# cp -n $MAG_RGAT_PATH/x_rgat_*.npy $MAG_MPLP_PATH/data/
# cp -n $MAG_FEAT_PATH/x_*.npy $MAG_MPLP_PATH/data/
# cp -n $MAG_FEAT_PATH/y_*.npy $MAG_MPLP_PATH/data/

# mkdir -p $MAG_MPLP_PATH/output

# python3 $MAG_CODE_PATH/mplp_ot.py \
#         $MAG_INPUT_PATH \
#         $MAG_BASE_PATH/dataset_path/mplp_data/feature/ \
#         $MAG_MPLP_PATH/output_temp/ \
#         --gpu \
#         --finetune \
#         --seed=0 \
#         --batch_size=100000 \
#         --epochs=200 \
#         --num_layers=2 \
#         --learning_rate=0.01 \
#         --dropout=0.5 \


python3 $MAG_CODE_PATH/mplp_ot.py \
        $MAG_INPUT_PATH \
        $MAG_BASE_PATH/dataset_path/mplp_data/feature/ \
        $MAG_MPLP_PATH/output_temp/ \
        --gpu \
        --finetune \
        --seed=0 \
        --batch_size=10000 \
        --epochs=200 \
        --num_layers=2 \
        --learning_rate=0.01 \
        --hidden=512 \
        --dropout=0.5 \


# mkdir -p $MAG_SUBM_PATH
# python3 $MAG_CODE_PATH/ensemble.py $MAG_INPUT_PATH $MAG_MPLP_PATH/output/ $MAG_SUBM_PATH

echo 'DONE!'
