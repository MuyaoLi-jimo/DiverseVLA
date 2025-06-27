import asyncio
from openai import AsyncOpenAI
import time
import json

# 替换为你的配置
my_key = "KEY"
my_url = "URL"
my_model = "MODEL"


# 这个函数处理单个请求，返回单个结果
async def async_query_openai(system_prompt, query, temperature=0.5, top_p=0.9, if_json=True):
    aclient = AsyncOpenAI(
        base_url=my_url,
        api_key=my_key  # API 密钥
    )
    # json格式
    if if_json:
        completion = await aclient.chat.completions.create(
            model=my_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            response_format={
                'type': 'json_object'
            },
            temperature=temperature,
            top_p=top_p,
        )
        return completion.choices[0].message.content
    # 非json格式
    else:
        completion = await aclient.chat.completions.create(
            model=my_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=temperature,
            top_p=top_p,
        )
        return completion.choices[0].message.content



# 这个函数接收一个请求列表，返回所有请求的结果列表
async def async_process_queries(system_prompt, queries, temperature=0.5, top_p=0.9, if_json=True):
    results = await asyncio.gather(*(async_query_openai(system_prompt, query, temperature=temperature, top_p=top_p, if_json=if_json) for query in queries))
    return results

# Str 转 json，失败返回 None
def str_to_json(r):
    try:
        json_r = json.loads(r)
        return json_r
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败", e)
        return None