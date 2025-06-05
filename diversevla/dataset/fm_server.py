
from openai import OpenAI
from anthropic import Anthropic
import os
import json
import requests
import multiprocessing as mp

from typing import Callable,List,Optional,Dict
from rich import print,console
from tqdm import tqdm
from time import sleep
import traceback
from colorama import Fore, Style
from pathlib import Path,Path
from typing import Literal,List,Union
import numpy as np
from PIL import Image

import base64
import httpx
from diversevla.utils import img_utils

ANTHROPIC_MODEL_LIST = {
    "claude-1",
    "claude-2",
    "claude-2.0",
    "claude-2.1",
    "claude-3-haiku-20240307",
    "claude-3-haiku-20240307-vertex",
    "claude-3-sonnet-20240229",
    "claude-3-sonnet-20240229-vertex",
    "claude-3-5-sonnet"
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-opus-20240229",
    "claude-instant-1",
    "claude-instant-1.2",
}

OPENAI_MODEL_LIST = {
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-browsing",
    "gpt-4-turbo-2024-04-09",
    "gpt2-chatbot",
    "im-also-a-good-gpt2-chatbot",
    "im-a-good-gpt2-chatbot",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o",
    "gpt-4o-mini",
    "chatgpt-4o-latest-20240903",
    "chatgpt-4o-latest",
    "o1-preview",
    "o1-mini",
}

DEEPSEEK_MODEL_LIST = {
    "deepseek-chat", #0.1元/百万tokens 2元/百万tokens 4K token
    "deepseek-reasoner",
}

PROPRIETART_MODEL_LIST =set()
PROPRIETART_MODEL_LIST.update(DEEPSEEK_MODEL_LIST,OPENAI_MODEL_LIST,ANTHROPIC_MODEL_LIST)

def create_system_message(system_prompt,model_type="jarvis-train"):
    message = None
    if model_type in OPENAI_MODEL_LIST or model_type in ANTHROPIC_MODEL_LIST or "qwen" in model_type:
        message = {
            "role": "system",
            "content": f"{system_prompt}\n",
        }
    elif model_type == "jarvis-train":
        message = {
            "role": "system",
            "content": [{
                "type": "text",
                "text": f"{system_prompt}\n"
            }]
        }
    else:
        raise ValueError(f"haven't define the model_id: {model_type}")
    return message

def create_assistant_message(input_type:Literal["text","bbox","point","reason"]="text",assistant_prompt="",source_data:List=None,model_type="jarvis-train"):
    message =  {
        "role": "assistant",
        "content": [],
    }
    if isinstance(assistant_prompt,str):
        assistant_prompt =  [assistant_prompt] if assistant_prompt else []
    elif not isinstance(assistant_prompt,list):
        raise ValueError(f"unset type: {type(assistant_prompt)}")
    if source_data is not None:
        assert isinstance(source_data,list)
    
    if input_type == "text":
        if model_type =="jarvis-train":
            for idx,text in enumerate(assistant_prompt):
                message["content"].append({
                    "type": "text",
                    "text": f"{text}"
                },)
            return message
        message["content"] = f"{assistant_prompt}"
    elif input_type == "bbox":
        source_num = len(source_data)
        if model_type =="jarvis-train":
            for idx,text in enumerate(assistant_prompt):
                if text:
                    message["content"].append({
                        "type": "text",
                        "text": f"{text}"
                    },)
                if idx<source_num:
                    message["content"].append({
                        "type": "bbox",
                        "bbox": source_data[idx]["bbox"],
                        "label": source_data[idx]["label"],
                    },)
            for idx in range(len(assistant_prompt), source_num):
                message["content"].append({
                    "type": "bbox",
                    "bbox": source_data[idx]["bbox"],
                    "label": source_data[idx]["label"],
                },)
    elif input_type == "point":
        source_num = len(source_data)

        if model_type =="jarvis-train":
            for idx,text in enumerate(assistant_prompt):
                if text:
                    message["content"].append({
                        "type": "text",
                        "text": f"{text}"
                    },)
                if idx<source_num:
                    message["content"].append({
                        "type": "point",
                        "point": source_data[idx]["point"],
                        "label": source_data[idx]["label"],
                    },)
            for idx in range(len(assistant_prompt), source_num):
                message["content"].append({
                    "type": "point",
                    "point": source_data[idx]["point"],
                    "label": source_data[idx]["label"],
                })
        else:
            raise ValueError(f"model_type unknown: {model_type}")
    elif input_type == "reason":
        assert len(source_data) == 1
        if model_type =="jarvis-train":
            message["content"].extend([{
                "type": "think",
                "text": f"{source_data[0]}"
            },{
                "type": "text",
                "text": f"{assistant_prompt[0]}"
            }])
        else:
            raise ValueError(f"model_type unknown: {model_type}")   
    return message

def get_suffix(image:Union[list,str,Path,np.ndarray,Image.Image]):
    if isinstance(image,np.ndarray|Image.Image):
        image_suffix = 'jpeg'
    elif isinstance(image,str):
        image = Path(image)
        image_suffix = image.suffix[1:]
    elif isinstance(image,Path):
        image_suffix = image.suffix[1:]
    else:
        raise ValueError(f"invalid image type！")
    return image_suffix

def get_image_message(source_data:Union[str,Path,np.ndarray,Image.Image],model_type:str):
    image_suffix = get_suffix(source_data)
    image = None
    if model_type == "transformers":
        image = f"data:image/{image_suffix};base64,{img_utils.encode_image_to_base64(source_data)}"
        image_message = {
            "type": "image_url",
            "image_url": image,
        }
    elif model_type in ANTHROPIC_MODEL_LIST:
        image_suffix = "jpeg" if image_suffix=="jpg" else image_suffix
        image_message = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": f"image/{image_suffix}",
                    "data": img_utils.encode_image_to_base64(source_data)
                },
            }
    else:
        image = { "url": f"data:image/{image_suffix};base64,{img_utils.encode_image_to_base64(source_data)}"}
        image_message = {
                "type": "image_url",
                "image_url": image,
            }
    return image_message

def create_user_message(input_type:Literal["text","image"], user_prompt:Union[str,List[str]]="",
                        source_data:Union[List[Union[str,Path,np.ndarray,Image.Image]],str,Path,np.ndarray,Image.Image]=[],
                        model_type:str="jarvis-train"):
    message = None
    if isinstance(user_prompt,str):
        user_prompt = [user_prompt]
    message = {
        "role": "user",
        "content": [],
    }
    if input_type=="text":
        for p in user_prompt:
            if p:
                message["content"].append({
                    "type": "text",
                    "text": f"{p}"
                })
    elif input_type=="image":
        if not isinstance(source_data ,list):
            source_data = [source_data]
        image_len = len(source_data)
        if model_type == "jarvis-train":
            for idx,text in enumerate(user_prompt):
                if text:
                    message["content"].append({
                        "type": "text",
                        "text": f"{text}"
                    },)
                if idx<image_len:
                    message["content"].append({
                        "type": "image",
                        "text": "<image>",
                    },)
            for idx in range(len(user_prompt), image_len):
                message["content"].append({
                    "type": "image",
                    "text": "<image>",
                },)
        elif model_type == "new-train":
            for idx,text in enumerate(user_prompt):
                if text:
                    message["content"].append({
                        "type": "text",
                        "text": f"{text}"
                    },)
                if idx<image_len:
                    message["content"].append({
                        "type": "image",
                    },)
            for idx in range(len(user_prompt), image_len):
                message["content"].append({
                    "type": "image",
                },)
        else:
            for idx, text in enumerate(user_prompt):
                if text:
                    message["content"].append({
                        "type": "text",
                        "text": f"{text}"
                    })
                if idx < image_len:
                    if model_type in ANTHROPIC_MODEL_LIST:
                        message["content"][-1]["text"] += "<image>"
                    message["content"].append(get_image_message(source_data[idx],model_type=model_type))
            for idx in range(len(user_prompt), image_len):
                if model_type in ANTHROPIC_MODEL_LIST:
                    message["content"][-1]["text"] += "<image>"
                message["content"].append(get_image_message(source_data[idx],model_type=model_type)) 
    else:
        raise AssertionError(f"type error:{input_type}")
    return message

def generate_responce(data, model_id = "gpt-4o",api_key="EMPTY", base_url="https://api.openai.com/v1/chat/completions", 
                temperature=0.7,max_tokens=4096,extra_body=None, **kwargs,):
    api_key = data.get("api_key",api_key)
    base_url = data.get("base_url",base_url)
    model = data.get("model_id",model_id)
    
    if model in ANTHROPIC_MODEL_LIST:
        client = Anthropic(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key, base_url=base_url)
    
    if not model:
        models = client.models.list()
        model = models.data[0].id
    temperature = data.get("temperature",temperature)
    max_tokens = data.get("max_tokens",max_tokens)
    input_extra_body = {"skip_special_tokens":False}
    if extra_body is not None:
        assert isinstance(extra_body,dict)
        input_extra_body.update(extra_body)
    response = None
    result = dict()
    try:
        if model in ANTHROPIC_MODEL_LIST:
            #print(data["messages"])
            if "thinking" in data:
                kwargs["thinking"] = data["thinking"]
                temperature = 1
            if data["messages"] and data["messages"][0].get("role","") == "system":
                data["messages"] = data["messages"][1:]
                kwargs["system"] = data["messages"][0]["content"]
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=data["messages"],
                **kwargs,
            )
            for block in response.content:
                if block.type == "thinking":
                    result["reasoning_content"] = block.thinking
                elif block.type == "text":
                    result["content"] = block.text
            result["token"] = {
                "input" : response.usage.input_tokens,
                "output": response.usage.output_tokens,
            }
        else:
            extra_body = input_extra_body,
            response = client.chat.completions.create(
                model=model,
                messages = data["messages"],
                temperature = temperature,
                max_tokens = max_tokens,
                extra_body = input_extra_body,
                **kwargs,
            )
            content = response.choices[0].message.content
            token = {
                "input" : response.usage.prompt_tokens,
                "output" : response.usage.completion_tokens,
            }

            result = {
                "content":content,
                "token":token,
            }
        
            if hasattr(response.choices[0].message,"reasoning_content"):
                result["reasoning_content"] = response.choices[0].message.reasoning_content
            result["all"] = response.choices[0]
    except Exception as e:
        print("response",response, e)
        raise AssertionError(f"id: {data.get('id')},error: {e}")
    
    return result