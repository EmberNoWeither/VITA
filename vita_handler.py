import torch
from ts.torch_handler.base_handler import BaseHandler
from vita.constants import IMAGE_TOKEN_INDEX
from vita.util.mm_utils import tokenizer_image_token, get_model_name_from_path
from vita.conversation import conv_templates, SeparatorStyle
from vita.util.mm_utils import KeywordsStoppingCriteria
from vita.util.utils import disable_torch_init
import json
import os

disable_torch_init()


class VITAHandler(BaseHandler):
    def __init__(self):
        super(VITAHandler, self).__init__()
        self.initialized = False
        self.gpu_counts = torch.cuda.device_count()
        os.environ['WORLD_SIZE'] = str(self.gpu_counts)

    def initialize(self, context):
        # 加载模型和相关组件
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        properties = context.system_properties
        
        
        self.map_location = "cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        
        model_dir = "/lpai/volumes/ss-nlu-ali-sh/lizr/xiaguoyang/checkpoints/vita-checkpoints/llava-s2-pretrain_video-multinode-stage3"

        model_path = model_dir  # 根据实际情况调整
        model_name = get_model_name_from_path(model_path)

        from vita.model.builder import load_pretrained_model
        with torch.device(self.device):
            self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model(
                model_path, None, model_name, model_type="mixtral-8x7b", device=self.device, only_one_gpu=True)

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(dtype=torch.float16, device=self.device)

        audio_encoder = self.model.get_audio_encoder()
        # audio_encoder.to(device="cuda", dtype=torch.float16)
        audio_encoder.to(dtype=torch.float16, device=self.device)
        audio_processor = audio_encoder.audio_processor

        # 添加设备同步检查
        print(f"Vision Tower device: {self.model.get_vision_tower().device}")
        print(f"Audio Encoder device: {next(self.model.get_audio_encoder().parameters()).device}")
        print(f"Main model device: {next(self.model.parameters()).device}")
        assert self.model.get_vision_tower().device == next(self.model.get_audio_encoder().parameters()).device == next(self.model.parameters()).device
            
        # self.model.to(self.device).half()
        self.model.eval()

        self.initialized = True

    def preprocess(self, data):
        # 处理输入数据
        request = data[0]['body']
        messages = request['messages']
        conv_mode = "mixtral_two"  
        conv = conv_templates[conv_mode].copy()
        for message in messages:
            if message['role'] == 'user':
                conv.append_message(conv.roles[0], message['content'])
            elif message['role'] == 'assistant':
                conv.append_message(conv.roles[1], message['content'])
                
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt(modality='lang')
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)

        # 停止标志
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        self.stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        return input_ids

    def inference(self, input_ids):
        image_tensor = torch.zeros((1, 3, 448, 448)).to(dtype=self.model.dtype, device=self.device).half()
        audio = torch.zeros(400, 80)
        audio_length = audio.shape[0]
        audio = torch.unsqueeze(audio, dim=0)
        audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
        audios = dict()
        audios["audios"] = audio.half().to(self.device)
        audios["lengths"] = audio_length.half().to(self.device)

        # 模型推理
        with torch.inference_mode(), torch.device(self.device):
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                audios=audios,
                do_sample=False,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[self.stopping_criteria])

        outputs = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        if '<3>' in outputs[:3]:
            outputs = outputs[3:]
        print(outputs)
        return [outputs]

    def postprocess(self, inference_output):
        # 处理输出结果
        # print(inference_output)
        # result = {'output': inference_output[0]}
        # return json.dumps(result)
        return inference_output