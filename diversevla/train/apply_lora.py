import argparse

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForVision2Seq
import os
import shutil

def apply_lora(base_model_path, target_model_path, lora_path):
    
    # base_model = AutoModelForVision2Seq.from_pretrained(
    #     base_model_path,
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # )
    
    # lora_model = PeftModel.from_pretrained(
    #     base_model,
    #     lora_path,
    #     # torch_dtype=torch.float16
    # )
    
    
    # print("Applying the LoRA")
    # model = lora_model.merge_and_unload(progressbar=True)
    
    # print(f"Saving the target model to {target_model_path}")
    # model.save_pretrained(target_model_path)
    
    files_to_copy = [
        'added_tokens.json',
        'README.md',
        'config.json',
        'special_tokens_map.json',
        'dataset_statistics.json',
        'tokenizer_config.json',
        'generation_config.json',
        'tokenizer.json',
        'preprocessor_config.json',
        'tokenizer.model'
    ]

    # 确保目标目录存在
    os.makedirs(target_model_path, exist_ok=True)
    
    # 复制文件
    for filename in files_to_copy:
        src_path = os.path.join(base_model_path, filename)
        dst_path = os.path.join(target_model_path, filename)
        try:
            shutil.copy2(src_path, dst_path)
            print(f"已复制: {filename}")
        except FileNotFoundError:
            print(f"未找到文件: {filename}")
        except Exception as e:
            print(f"复制 {filename} 时出错: {e}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, default="/share/lmy/models/openvla-7b-finetuned-libero-10")
    parser.add_argument("--lora-path", type=str, default="/home/lmy/workspace/DiverseVLA/checkpoints/openvla-7b-finetuned-libero-10+libero_10_no_noops+b16+lr-5e-06+lora-r64+dropout-0.0--image_aug")
    #注意不要把lora加进去
    parser.add_argument("--target-model-path", type=str, default="/share/lmy/models/trained-openvla-7b/openvla-7b-finetuned-libero-10")
    
    args = parser.parse_args()

    apply_lora(args.base_model_path,  args.target_model_path, args.lora_path)