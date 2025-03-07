#!/bin/bash


CONV="mixtral_two"
MODEL_TYPE="mixtral-8x7b"
CKPT_NAME="VITA-BASELINE"
CKPT="/lpai/volumes/ss-nlu-ali-sh/lizr/xiaguoyang/checkpoints/vita-checkpoints/llava-s2-pretrain_video-multinode-stage3"
EVAL="/lpai/volumes/ss-nlu-ali-sh/lizr/xiaguoyang/moellava_data/llava_eval"

CUDA_VISIBLE_DEVICES=0,1 python3  vita/eval/model_vqa_loader.py \
    --model-path ${CKPT} \
    --model_type ${MODEL_TYPE} \
    --question-file ${EVAL}/pope/llava_pope_test.jsonl \
    --image-folder ${EVAL}/pope/val2014 \
    --answers-file ${EVAL}/pope/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv_mode ${CONV}

python3 vita/eval/eval_pope.py \
    --annotation-dir ${EVAL}/pope/coco \
    --question-file ${EVAL}/pope/llava_pope_test.jsonl \
    --result-file ${EVAL}/pope/answers/${CKPT_NAME}.jsonl