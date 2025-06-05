from dataclasses import dataclass
import draccus
from typing import Literal
from pathlib import Path
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

def get_tasks(libero_suite_name:Literal["libero_object"]="libero_object"):
    return libero_task_map[libero_suite_name]

def get_instruction(task:str):
    instruction = " ".join(task.split("_"))
    return instruction


@dataclass
class GenerateDiverseConfig:
    libero_suite_name:str = "libero_object"
    change_scope: Literal["one", "two", "all"] = "one"
    output_path:str = f""
    
    model:str = "gpt-4o"
    
    def __post_init__(self):
        self.output_path = Path(f"./data/{self.libero_suite_name}/{self.change_scope}")
        self.output_path.mkdir(parents=True,exist_ok=True)
   
@draccus.wrap() 
def revise_datasets(cfg:GenerateDiverseConfig):
    tasks = get_tasks(cfg.libero_suite_name)
    instruction_dict = { task:get_instruction(task) for task in tasks}
    instruction_list = [ {task:instruction} for task,instruction in instruction_dict.items()]
    


if __name__ == "__main__":
    
    
    pass