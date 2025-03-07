#!/bin/bash

MODEL_TYPE=mixtral-8x7b
OUTPUT_DIR=$1
OUTPUT_DIR_FT=${OUTPUT_DIR}/llava-s2-pretrain_video
mkdir -p ${OUTPUT_DIR_FT}

deepspeed --include localhost:0,1,2,3,4,5,6,7 vita/train/train.py \
    --deepspeed ./script/deepspeed/ds_config_zero3_offload2.json \
    --model_name_or_path /mnt/pfs-mc0p4k/nlu/team/xiaguoyang/models/mistralai-Mixtral-8x7B-v0.1 \
    --model_type $MODEL_TYPE \
    --version mixtral_two \
    --dataset_use MoE_LLaVA_S2 \
    --vision_tower /mnt/pfs-mc0p4k/nlu/team/xiaguoyang/models/OpenGVLab-InternViT-300M-448px \
    --pretrain_mm_mlp_adapter ${OUTPUT_DIR}/llava-s1-pretrain_mlp_video/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --audio_encoder /mnt/pfs-mc0p4k/nlu/team/xiaguoyang/models/VITA-MLLM-VITA/audio-encoder-2wh_zh_en_audioset_Mixtral-8x7B_New-base-tunning \
    --freeze_audio_encoder True \
    --freeze_audio_encoder_adapter False \
    --image_folder "/mnt/pfs-mc0p4k/nlu/team/xiaguoyang/data/LLaVA-MoE-Data" \
    --image_aspect_ratio square \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR_FT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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

