from dataclasses import dataclass
import draccus
from typing import Literal
from pathlib import Path
from functools import partial
from libero.libero.benchmark.libero_suite_task_map import libero_task_map
from diversevla.dataset.revise import revise_instruct
from diversevla.utils import img_utils,file_utils,mp_utils

def get_tasks(libero_suite_name:Literal["libero_object"]="libero_object"):
    return libero_task_map[libero_suite_name]

def get_instruction(task:str):
    instruction = " ".join(task.split("_"))
    return instruction


@dataclass
class GenerateDiverseConfig:
    libero_suite_name:str = "libero_object"
    change_scope: Literal["one", "two", "all"] = "all"
    output_path:str = f""
    
    model:str = "gpt-4o"
    
    def __post_init__(self):
        self.output_path = Path(f"./data/{self.libero_suite_name}/{self.change_scope}/instruction.json")
        self.output_path.parent.mkdir(parents=True,exist_ok=True)
   
@draccus.wrap() 
def revise_datasets(cfg:GenerateDiverseConfig):
    tasks = get_tasks(cfg.libero_suite_name)
    instruction_dict = { task:get_instruction(task) for task in tasks}
    instruction_list = [ {"task":task,"instruction":instruction} for task,instruction in instruction_dict.items()]
    revise_instruct_wrapper = partial(revise_instruct,change_scope=cfg.change_scope)
    datas = mp_utils.get_multiple_response(revise_instruct_wrapper,input_datas=instruction_list,)
    data_dict = { data["task"]:data["new_instroduction"] for data in datas}
    file_utils.dump_json_file(datas,cfg.output_path)


if __name__ == "__main__":
    revise_datasets()
