#!/bin/bash
CONV="mixtral_two"
MODEL_TYPE="mixtral-8x7b"
CKPT_NAME="VITA-BASELINE"
CKPT="/lpai/volumes/ss-nlu-ali-sh/lizr/xiaguoyang/checkpoints/vita-checkpoints/llava-s2-pretrain_video-multinode-stage3"
EVAL="/lpai/volumes/ss-nlu-ali-sh/lizr/xiaguoyang/moellava_data/llava_eval"
CUDA_VISIBLE_DEVICES=0,1 python3  vita/eval/model_vqa_loader.py \
    --model-path ${CKPT} \
    --model_type ${MODEL_TYPE} \
    --question-file ${EVAL}/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ${EVAL}/textvqa/train_images \
    --answers-file ${EVAL}/textvqa/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv_mode ${CONV}

python3 -m vita/eval/eval_textvqa.py \
    --annotation-file ${EVAL}/textvqa/TextVQA_0.5.1_val.json \
    --result-file ${EVAL}/textvqa/answers/${CKPT_NAME}.jsonl