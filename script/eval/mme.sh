#!/bin/bash

CONV="mixtral_two"
MODEL_TYPE="mixtral-8x7b"
CKPT_NAME="VITA-BASELINE"
CKPT="/lpai/volumes/ss-nlu-ali-sh/lizr/xiaguoyang/checkpoints/vita-checkpoints/llava-s2-pretrain_video-multinode-stage3"
EVAL="/lpai/volumes/ss-nlu-ali-sh/lizr/xiaguoyang/moellava_data/llava_eval"
CUDA_VISIBLE_DEVICES=0,1 python3 vita/eval/model_vqa_loader.py \
    --model-path ${CKPT} \
    --model_type ${MODEL_TYPE} \
    --question-file ${EVAL}/MME/llava_mme.jsonl \
    --image-folder ${EVAL}/MME/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file ${EVAL}/MME/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv_mode ${CONV}

cd ${EVAL}/MME

python convert_answer_to_mme.py --experiment $CKPT_NAME

cd eval_tool

python calculation.py --results_dir answers/$CKPT_NAME

