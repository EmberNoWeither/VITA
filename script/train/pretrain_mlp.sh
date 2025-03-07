#!/bin/bash

MODEL_TYPE=mixtral-8x7b
OUTPUT_DIR=/mnt/pfs-mc0p4k/nlu/team/xiaguoyang/checkpoints/vita-checkpoints
OUTPUT_DIR_FT=${OUTPUT_DIR}/llava-s1-pretrain_mlp_video
mkdir -p ${OUTPUT_DIR_FT}

deepspeed vita/train/train.py \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path /mnt/pfs-mc0p4k/nlu/team/xiaguoyang/models/mistralai-Mixtral-8x7B-v0.1 \
    --model_type $MODEL_TYPE \
    --version mixtral_two \
    --dataset_use MoE_LLaVA_S1 \
    --vision_tower /mnt/pfs-mc0p4k/nlu/team/xiaguoyang/models/OpenGVLab-InternViT-300M-448px \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --audio_encoder /mnt/pfs-mc0p4k/nlu/team/xiaguoyang/models/VITA-MLLM-VITA/audio-encoder-2wh_zh_en_audioset_Mixtral-8x7B_New-base-tunning \
    --freeze_audio_encoder True \
    --freeze_audio_encoder_adapter True \
    --image_folder "/mnt/pfs-mc0p4k/nlu/team/xiaguoyang/data/LLaVA-MoE-Data" \
    --image_aspect_ratio square \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR_FT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 6200 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee -a ${OUTPUT_DIR_FT}/log.txt && echo "Done."


