import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from vita.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

from vita.conversation import conv_templates, SeparatorStyle
from vita.model.builder import load_pretrained_model
from vita.util.utils import disable_torch_init
from vita.util.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from vita.util.data_utils_video_audio_neg_patch import dynamic_preprocess
from torch.utils.data import Dataset, DataLoader
from vita.util.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_audio_token,
    tokenizer_image_token,
)
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image, p_num = dynamic_preprocess(
                    image, min_num=1, max_num=12, image_size=448, use_thumbnail=True
                )
        assert len(p_num) == 1, "Only one image is allowed"
        
        image_tensor = process_images(image, self.image_processor, self.model_config)
        # print(image_tensor.shape)
        
        qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + qs
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt(modality='image')

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, args.model_base,
                                                     model_name, args.model_type, only_one_gpu=args.only_one_gpu)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    audio_encoder = model.get_audio_encoder()
    # audio_encoder.to(device="cuda", dtype=torch.float16)
    audio_encoder.to(dtype=torch.float16)
    audio_processor = audio_encoder.audio_processor

    model.eval()

    audio = torch.zeros(400, 80)
    audio_length = audio.shape[0]
    audio = torch.unsqueeze(audio, dim=0)
    audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
    audios = dict()
    audios["audios"] = audio.half().cuda()
    audios["lengths"] = audio_length.half().cuda()

    if args.return_gating_logit is not None:
        from vita.utils import get_gating_logit_by_hook
        print(model)
        fea_hooks = get_gating_logit_by_hook(model)
        all_gating_logits = {}

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    cnt = -1
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        cnt += 1
        # if cnt == 30:
        #     break
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        conv = conv_templates[args.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.squeeze(0).half().cuda(),
                audios=audios,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True if args.return_gating_logit is None else False,
                stopping_criteria=stopping_criteria
            )
        if args.return_gating_logit is not None:
            # import ipdb
            # ipdb.set_trace()
            all_gating_logits[cnt] = dict(gating_logit=[i.fea for i in fea_hooks],
                                          images=image_tensor if image_tensor is None else image_tensor.detach().cpu(),
                                          input_ids=input_ids.detach().cpu(),
                                          output_ids=output_ids.detach().cpu())
            print(input_ids.shape, output_ids.shape, fea_hooks[0].fea.shape, image_tensor.shape if image_tensor is not None else [])
            # assert fea_hooks[0].fea.shape[0] + 1 == output_ids.shape[1] + 575
            print('The number of hooks is:', len(fea_hooks))

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        if '<3>' in outputs:
            outputs = outputs[3:]
        print(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

    if args.return_gating_logit is not None:
        torch.save(all_gating_logits, f'{args.return_gating_logit}.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="mixtral-8x7b")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--only_one_gpu", action="store_true", default=False)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv_mode", type=str, default="mixtral_two")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--return_gating_logit", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)
