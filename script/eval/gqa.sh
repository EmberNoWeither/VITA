#!/bin/bash

# 定义GPU组
GPU_GROUPS=(
  "0,1"
  "2,3"
  "4,5"
  "6,7"
)

# 组数量即为chunks数量
CHUNKS=${#GPU_GROUPS[@]}

# 模型配置
MODEL_TYPE="mixtral-8x7b"
CKPT_NAME="VITA-BASELINE"
CKPT="/lpai/volumes/ss-nlu-ali-sh/lizr/xiaguoyang/checkpoints/vita-checkpoints/llava-s2-pretrain_video-multinode-stage3"
CONV="mixtral_two"

# 评估路径
EVAL="/lpai/volumes/ss-nlu-ali-sh/lizr/xiaguoyang/moellava_data/llava_eval"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="${EVAL}/gqa/data"

# 创建输出目录
mkdir -p ${EVAL}/gqa/answers/$SPLIT/${CKPT_NAME}

# 为每个chunk并行运行评估
for IDX in $(seq 0 $((CHUNKS-1))); do
    # 使用预定义的GPU组
    CUDA_VISIBLE_DEVICES=${GPU_GROUPS[$IDX]} python3 vita/eval/model_vqa_loader.py \
        --model-path ${CKPT} \
        --model_type ${MODEL_TYPE} \
        --question-file ${EVAL}/gqa/$SPLIT.jsonl \
        --image-folder ${EVAL}/gqa/data/images \
        --answers-file ${EVAL}/gqa/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv_mode ${CONV} &
done

wait

# 合并结果
output_file=${EVAL}/gqa/answers/$SPLIT/${CKPT_NAME}/merge.jsonl

# 如果输出文件存在，清空它
> "$output_file"

# 循环遍历索引并连接每个文件
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${EVAL}/gqa/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# 转换和评估
mkdir -p $GQADIR/$SPLIT/${CKPT_NAME}
python3 script/eval/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/$SPLIT/${CKPT_NAME}/testdev_balanced_predictions.json

cd $GQADIR
python3 eval_gqa.py --tier $SPLIT/${CKPT_NAME}/testdev_balanced \
                    --questions ${EVAL}/gqa/data/testdev_balanced_questions.json