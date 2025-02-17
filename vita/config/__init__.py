from .dataset_config import *

NaturalCap = [ShareGPT4V]

MoE_LLaVA_S1 = [llava_image_]
MoE_LLaVA_S2 = [lvis, nlp, svit, lrv, la]
MoE_LLaVA_S3 = [llava_image_tune, nlp]

DataConfig = {
    "Pretrain_video": NaturalCap,
    'MoE_LLaVA_S1': MoE_LLaVA_S1,
    "MoE_LLaVA_S2": MoE_LLaVA_S2,
    'MoE_LLaVA_S3': MoE_LLaVA_S3,
}

NoPatchSets = ["khair", "jester"]
