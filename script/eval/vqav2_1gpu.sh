#!/bin/bash

# 定义GPU组
GPU_GROUPS=(
  "0"
  "1"
  "2"
  "3"
  "4"
  "5"
  "6"
  "7"
)

# 组数量即为chunks数量
CHUNKS=${#GPU_GROUPS[@]}

# 模型配置和评估路径
CONV="mixtral_two"
MODEL_TYPE="mixtral-8x7b"
CKPT_NAME="VITA-BASELINE"
CKPT="/lpai/volumes/ss-nlu-ali-sh/lizr/xiaguoyang/checkpoints/vita-checkpoints/llava-s2-pretrain_video-multinode-stage3"
SPLIT="llava_vqav2_mscoco_test-dev2015"
EVAL="/lpai/volumes/ss-nlu-ali-sh/lizr/xiaguoyang/moellava_data/llava_eval"

# 创建输出目录
mkdir -p ${EVAL}/vqav2/answers/$SPLIT/${CKPT_NAME}

# 为每个组运行评估
for IDX in $(seq 0 $((CHUNKS-1))); do
    # 使用指定的GPU组
    CUDA_VISIBLE_DEVICES=${GPU_GROUPS[$IDX]} python3 vita/eval/model_vqa_loader.py \
        --model-path ${CKPT} \
        --model_type ${MODEL_TYPE} \
        --question-file ${EVAL}/vqav2/$SPLIT.jsonl \
        --image-folder ${EVAL}/vqav2/test2015 \
        --only_one_gpu \
        --answers-file ${EVAL}/vqav2/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv_mode ${CONV} &
done

wait

output_file=${EVAL}/vqav2/answers/$SPLIT/${CKPT_NAME}/merge.jsonl

# 清空输出文件（如果存在）
> "$output_file"

# 循环遍历索引并连接每个结果文件
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${EVAL}/vqav2/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# 转换结果为提交格式
python3 script/eval/convert_vqav2_for_submission.py --split $SPLIT --ckpt ${CKPT_NAME} --dir ${EVAL}/vqav2