#!/usr/bin/env bash
export DGLBACKEND=pytorch
export CUDA_VISIBLE_DEVICES=2

MAG_BASE_PATH=/data/zhaohuanjing/mag/ # modify to your workspace

MAG_INPUT_PATH=$MAG_BASE_PATH/dataset_path/  # The MAG240M-LSC dataset should be placed here

MAG_CODE_PATH=$MAG_BASE_PATH/mplp/
MAG_PREP_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/preprocess/
# MAG_RGAT_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/rgat/
# MAG_FEAT_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/feature/
MAG_MPLP_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/mplp/
# MAG_SUBM_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/submission/


# !!!!!!!! NOTICE
# # !!! Please change the filename "outputs/try" for each run if you want to save the result. !!!

MAG_OUTPUT_PATH=$MAG_MPLP_PATH/outputs/rgat1024_label_m2v

python3 $MAG_CODE_PATH/mplp_folds.py \
        $MAG_INPUT_PATH \
        $MAG_BASE_PATH/dataset_path/mplp_data/feature/ \
        $MAG_OUTPUT_PATH \
        --gpu \
        --finetune \
        --wandb \
        --seed=0 \
        --batch_size=10000 \
        --epochs=200 \
        --num_layers=2 \
        --learning_rate=0.01 \
        --mlp_hidden=512 \
        --dropout=0.5 \
        --hidden=256 \

# POST_SMOOTHING_GRAPH_PATH=$MAG_PREP_PATH/ppgraph_cite.bin
# POST_SMOOTHING_GRAPH_PATH=$MAG_PREP_PATH/paper_coauthor_paper_symmetric_jc0.5.bin
# python3 $MAG_CODE_PATH/post_s.py $MAG_INPUT_PATH $MAG_OUTPUT_PATH $POST_SMOOTHING_GRAPH_PATH


# MAG_SUBM_PATH=$MAG_OUTPUT_PATH/subm
# MAG_METHODS_PATH=$MAG_MPLP_PATH/outputs
# # mkdir -p $MAG_SUBM_PATH
# python3 $MAG_CODE_PATH/ensemble_folds.py $MAG_INPUT_PATH $MAG_OUTPUT_PATH $MAG_SUBM_PATH
# python3 $MAG_CODE_PATH/ensemble_last.py $MAG_INPUT_PATH $MAG_METHODS_PATH $MAG_SUBM_PATH

echo 'DONE!'
