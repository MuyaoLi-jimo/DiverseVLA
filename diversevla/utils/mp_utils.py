from typing import Callable
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor

from tqdm import tqdm
from time import sleep
from rich import console,print
from typing import Union
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent ))

from diversevla.utils import file_utils

def create_chunk_responces_old(wrapper:Callable[[list],list],datas:list):
    raise AssertionError("作废")
    num_processes = len(datas)
    results = []
    if num_processes > 1:
        pool = mp.Pool(processes=num_processes) 
        #print(datas)  
        results = pool.map(wrapper, datas)  
        pool.close() # close the pool
        pool.join()
    else:
        results = [wrapper(datas[0])]
    return results

def create_chunk_responces(wrapper:Callable,datas:list):
    """
    This function manages the execution of a wrapper function over a list of data in parallel using multiprocessing.

    Args:
        wrapper (Callable[[list], list]): A function that processes a list of data.
        datas (list): A list containing data to be processed in chunks.

    Returns:
        list: A list of results after processing each data chunk.
    """
    results = []
    max_workers = min(len(datas), 128)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(wrapper, datas))
    return results

def thread_exe(wrapper: Callable, datas: list):
    max_workers = min(len(datas), 128)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(wrapper, datas))
    return results

def create_chunk_datas(input_datas:list,batch_size:int=80):
    chunk_size = len(input_datas)//batch_size if len(input_datas) % batch_size==0 else len(input_datas)//batch_size + 1
    chunks = []
    for i in tqdm(range(chunk_size)):
        if (i+1) * batch_size <= len(input_datas):
            chunk_datas = input_datas[i * batch_size: (i + 1) * batch_size]
        else:
            chunk_datas = input_datas[i*batch_size:len(input_datas)]
        chunks.append(chunk_datas)
    return chunks

def get_multiple_response(wrapper:Callable[[list],list],
                          input_datas:list,
                          batch_size = 80,
                          store_fold_path:Union[str,Path] = None,
                          slow=False,
                          max_wait_time:int=1,
                          flag_id="id",
                          if_backup=True,
                          collect_all=True):
    output_datas = []
    
    # reload_data
    if store_fold_path:
        temp_jp = file_utils.JsonlProcessor(store_fold_path,if_backup=if_backup)
        
        output_datas = temp_jp.load_lines()
        
        if flag_id:
            reload_set = set()
            for output in output_datas:
                reload_set.add(output[flag_id])
                
            reload_datas = []
            for input_data in input_datas:
                if input_data[flag_id] not in reload_set:
                    reload_datas.append(input_data)
            input_datas = reload_datas
            print(f"断点续连: 这个阶段还要有{len(input_datas)}")
        
    chunk_size = len(input_datas)//batch_size if len(input_datas) % batch_size==0 else len(input_datas)//batch_size + 1
    for i in tqdm(range(chunk_size)):
        if (i+1) * batch_size <= len(input_datas):
            chunk_datas = input_datas[i * batch_size: (i + 1) * batch_size]
        else:
            chunk_datas = input_datas[i*batch_size:len(input_datas)]
        
        
        for i in range(max_wait_time):
            try:
                chunk_results = create_chunk_responces(wrapper,chunk_datas)
                if slow:
                    sleep(15)
                break
            except Exception as e:
                console.Console().log(f"[red]fail to run the wrapper!,caused by {e}")
                chunk_results = []
                if slow:
                    sleep(120+i*120)
                
        if not chunk_results:
            raise ValueError("if not chunk_results:",{chunk_results})
        
        new_chunk_results = []
        # !我们认为每个输出返回的是一个list
        for chunk_result in chunk_results:
            new_chunk_results.extend(chunk_result)
        if collect_all:
            output_datas.extend(new_chunk_results)
        if store_fold_path:
            temp_jp.dump_lines(new_chunk_results)
    if store_fold_path:
        temp_jp.close()
    return output_datas