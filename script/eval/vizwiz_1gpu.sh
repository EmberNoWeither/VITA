#!/bin/bash
CONV="mixtral_two"
MODEL_TYPE="mixtral-8x7b"
CKPT_NAME="VITA-BASELINE"
CKPT="/lpai/volumes/ss-nlu-ali-sh/lizr/xiaguoyang/checkpoints/vita-checkpoints/llava-s2-pretrain_video-multinode-stage3"
EVAL="/lpai/volumes/ss-nlu-ali-sh/lizr/xiaguoyang/moellava_data/llava_eval"
CUDA_VISIBLE_DEVICES=7 python3 vita/eval/model_vqa_loader.py \
    --model-path ${CKPT} \
    --model_type ${MODEL_TYPE} \
    --question-file ${EVAL}/vizwiz/llava_test.jsonl \
    --image-folder ${EVAL}/vizwiz/test \
    --only_one_gpu \
    --answers-file ${EVAL}/vizwiz/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv_mode ${CONV}

python3 script/eval/convert_vizwiz_for_submission.py \
    --annotation-file ${EVAL}/vizwiz/llava_test.jsonl \
    --result-file ${EVAL}/vizwiz/answers/${CKPT_NAME}.jsonl \
    --result-upload-file ${EVAL}/vizwiz/answers_upload/${CKPT_NAME}.json