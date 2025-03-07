#!/bin/bash

SPLIT="mmbench_dev_20230712"
CONV="mixtral_two"
MODEL_TYPE="mixtral-8x7b"
CKPT_NAME="VITA-BASELINE"
CKPT="/lpai/volumes/ss-nlu-ali-sh/lizr/xiaguoyang/checkpoints/vita-checkpoints/llava-s2-pretrain_video-multinode-stage3"
EVAL="/lpai/volumes/ss-nlu-ali-sh/lizr/xiaguoyang/moellava_data/llava_eval"

CUDA_VISIBLE_DEVICES=0,1 python3 vita/eval/model_vqa_mmbench.py \
    --model-path ${CKPT} \
    --model_type ${MODEL_TYPE} \
    --question-file ${EVAL}/mmbench/$SPLIT.tsv \
    --answers-file ${EVAL}/mmbench/answers/$SPLIT/${CKPT_NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode ${CONV} 

mkdir -p ${EVAL}/mmbench/answers_upload/$SPLIT

python3 script/eval/convert_mmbench_for_submission.py \
    --annotation-file ${EVAL}/mmbench/$SPLIT.tsv \
    --result-dir ${EVAL}/mmbench/answers/$SPLIT \
    --upload-dir ${EVAL}/mmbench/answers_upload/$SPLIT \
    --experiment ${CKPT_NAME}