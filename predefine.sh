mkdir -p /opt/conda/envs/vita &&

tar -xvf /mnt/pfs-mc0p4k/nlu/team/xiaguoyang/envs/vita.tar.gz -C /opt/conda/envs/vita &&

/opt/conda/envs/vita/bin/conda-unpack &&

export PATH=/opt/conda/envs/vita/bin:$PATH &&

conda init bash &&

source ~/.bashrc &&

conda activate vita 

# bash /mnt/pfs-mc0p4k/nlu/team/xiaguoyang/code/MoE-LLaVA/scripts/v1/phi2/finetune.sh

cd /mnt/pfs-mc0p4k/nlu/team/xiaguoyang/code/VITA

export PYTHONPATH=$PYTHONPATH:/mnt/pfs-mc0p4k/nlu/team/xiaguoyang/code/VITA

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# bash /mnt/pfs-mc0p4k/nlu/team/xiaguoyang/code/MoE-LLaVA/scripts/v1/phi2/finetune_moe_stage2_softmodal.sh
bash /mnt/pfs-mc0p4k/nlu/team/xiaguoyang/code/VITA/script/train/pretrain_mlp.sh
