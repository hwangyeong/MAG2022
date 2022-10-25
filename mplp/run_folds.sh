#!/usr/bin/env bash
export DGLBACKEND=pytorch
export CUDA_VISIBLE_DEVICES=5

MAG_BASE_PATH=/data/zhaohuanjing/mag/ # modify to your workspace

MAG_INPUT_PATH=$MAG_BASE_PATH/dataset_path/  # The MAG240M-LSC dataset should be placed here

MAG_CODE_PATH=$MAG_BASE_PATH/mplp/
# MAG_PREP_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/preprocess/
# MAG_RGAT_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/rgat/
# MAG_FEAT_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/feature/
MAG_MPLP_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/mplp/
# MAG_SUBM_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/submission/




# !!!!!!!! NOTICE
# # !!! Please change the filename "outputs/try" for each run if you want to save the result. !!!

MAG_OUTPUT_PATH=$MAG_MPLP_PATH/outputs/try

python3 $MAG_CODE_PATH/mplp_folds.py \
        $MAG_INPUT_PATH \
        $MAG_BASE_PATH/dataset_path/mplp_data/feature/ \
        $MAG_OUTPUT_PATH \
        --gpu \
        --finetune \
        --seed=0 \
        --batch_size=100000 \
        --epochs=200 \
        --num_layers=2 \
        --learning_rate=0.01 \
        --mlp_hidden=512 \
        --dropout=0.5 \


# mkdir -p $MAG_SUBM_PATH
# python3 $MAG_CODE_PATH/ensemble.py $MAG_INPUT_PATH $MAG_MPLP_PATH/output/ $MAG_SUBM_PATH
MAG_SUBM_PATH=$MAG_OUTPUT_PATH/subm
mkdir -p $MAG_SUBM_PATH
python3 $MAG_CODE_PATH/ensemble_folds.py $MAG_INPUT_PATH $MAG_OUTPUT_PATH $MAG_SUBM_PATH

echo 'DONE!'
