from OpenVLA.prismatic.vla.datasets.datasets import RLDSDataset
import tensorflow_datasets as tfds
from tensorflow_datasets.core.example_serializer import ExampleSerializer
import tensorflow as tf
import dlimp as dl
import json
from pathlib import  Path
from diversevla.utils import file_utils
import os
import random
import traceback
import glob
from tqdm import tqdm

def extract_instructions_to_json(dataset_name, data_dir, split="train", output_dir="./assets/"):
    output_path = Path(output_dir) / dataset_name / "instruction.json"
    output_path.parent.mkdir(exist_ok=True,parents=True)
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    builder.download_and_prepare()
    ds = builder.as_dataset(split=split, shuffle_files=False)

    instructions = {}
    for i, episode in enumerate(tfds.as_numpy(ds)):
        step_dataset = episode["steps"]
        # 获取第一步中的 language_instruction
        try:
            first_step = next(iter(step_dataset))
            raw_instr = first_step["language_instruction"]
            if isinstance(raw_instr, bytes):
                decoded_instr = raw_instr.decode("utf-8")
            else:
                decoded_instr = str(raw_instr)
            instructions[str(i)] = decoded_instr
        except Exception as e:
            print(f"[Warning] Episode {i} has no instruction. Skipped. Error: {e}")
            continue

    # 保存为 JSON 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(instructions, f, indent=2, ensure_ascii=False)

    print(f"✅ 提取完成，共保存 {len(instructions)} 条指令到: {output_path}")

def accumulate_instructions(dataset_name, base_instruction_dir="./assets/"):
    old_instruction_path = Path(base_instruction_dir) / dataset_name / "instruction.json"
    instruction_accumulate_path = Path(base_instruction_dir) / dataset_name / "instruction_accumulate.json"
    old_instructions = file_utils.load_json_file(old_instruction_path)
    instructions_dict = {}
    for k,old_instruction in old_instructions.items():
        same_instruction_list = instructions_dict.get(old_instruction,[k])
        same_instruction_list.append(k)
        instructions_dict[old_instruction] = same_instruction_list
    file_utils.dump_json_file(instructions_dict,instruction_accumulate_path)

def get_restruct_instructions(dataset_name,base_instruction_dir="./assets/"):
    instruction_dir = Path(base_instruction_dir) / dataset_name
    polish_name = "_".join(dataset_name.split("_")[:2])
    instruction_paths = [ instruction_dir / f"{name}_{polish_name}.json" for name in ["one","two","all"]] 
    instruction_dict = {}
    for instruction_path in instruction_paths:
        partial_instruction_dict = file_utils.load_json_file(instruction_path)
        for k,v in partial_instruction_dict.items():
            if k not in instruction_dict:
                instruction_dict[k] = v
            else:
                instruction_dict[k].extend(v)
    return instruction_dict


# --- 辅助函数 (保持不变，但要注意其适用范围) ---
def _bytes_feature(value):
    """Returns a bytes_list from a SINGLE string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# --- 最终修正版函数 ---
def replace_instructions_and_save_low_level(
    dataset_name,
    data_dir,
    split,
    new_dir,
    new_name
):
    replace_map = get_restruct_instructions(dataset_name)

    try:
        builder_for_info = tfds.builder(dataset_name, data_dir=data_dir)
        version_dir = builder_for_info.info.version
        file_pattern = os.path.join(data_dir, dataset_name, str(version_dir), f'*-{split}.tfrecord*')
        source_files = glob.glob(file_pattern)
        if not source_files:
            raise FileNotFoundError(f"No TFRecord files found for pattern: {file_pattern}")
        print(f"Found {len(source_files)} source TFRecord files: {source_files[0]} ...")
    except Exception as e:
        print(f"Could not automatically find files, please specify the path manually. Error: {e}")
        return

    feature_description = {
        'episode_metadata/file_path': tf.io.FixedLenFeature([], tf.string),
        'steps/action': tf.io.VarLenFeature(tf.float32),
        'steps/discount': tf.io.VarLenFeature(tf.float32),
        'steps/is_first': tf.io.VarLenFeature(tf.int64),
        'steps/is_last': tf.io.VarLenFeature(tf.int64),
        'steps/is_terminal': tf.io.VarLenFeature(tf.int64),
        'steps/language_instruction': tf.io.VarLenFeature(tf.string),
        'steps/reward': tf.io.VarLenFeature(tf.float32),
        'steps/observation/image': tf.io.VarLenFeature(tf.string),
        'steps/observation/joint_state': tf.io.VarLenFeature(tf.float32),
        'steps/observation/state': tf.io.VarLenFeature(tf.float32),
        'steps/observation/wrist_image': tf.io.VarLenFeature(tf.string),
    }

    raw_dataset = tf.data.TFRecordDataset(source_files)
    os.makedirs(new_dir, exist_ok=True)
    output_path = os.path.join(new_dir, new_name)
    writer = tf.io.TFRecordWriter(output_path)

    for i, raw_record in tqdm(enumerate(raw_dataset)):
        try:
            parsed_episode = tf.io.parse_single_example(raw_record, feature_description)
            num_steps = parsed_episode['steps/is_first'].values.shape[0]
            if num_steps == 0:
                continue

            raw_instr = parsed_episode['steps/language_instruction'].values[0].numpy()
            old_instr = raw_instr.decode("utf-8") if isinstance(raw_instr, bytes) else str(raw_instr)
            new_instr = random.choice(replace_map.get(old_instr, [old_instr]))
            new_instr_bytes = new_instr.encode('utf-8')

            # 手动重建 Feature 字典用于写入
            feature_dict = {
                # _bytes_feature 在这里使用是正确的，因为 file_path 是单个值
                'episode_metadata/file_path': _bytes_feature(parsed_episode['episode_metadata/file_path'].numpy()),

                # <-- 最终改动在这里 -->
                # 对于字节列表，我们直接构造 Feature，不使用有问题的辅助函数
                'steps/language_instruction': tf.train.Feature(bytes_list=tf.train.BytesList(value=tf.tile(tf.constant([new_instr_bytes]), [num_steps]).numpy())),

                'steps/action': tf.train.Feature(float_list=tf.train.FloatList(value=parsed_episode['steps/action'].values)),
                'steps/discount': tf.train.Feature(float_list=tf.train.FloatList(value=parsed_episode['steps/discount'].values)),
                'steps/is_first': tf.train.Feature(int64_list=tf.train.Int64List(value=parsed_episode['steps/is_first'].values.numpy())),
                'steps/is_last': tf.train.Feature(int64_list=tf.train.Int64List(value=parsed_episode['steps/is_last'].values.numpy())),
                'steps/is_terminal': tf.train.Feature(int64_list=tf.train.Int64List(value=parsed_episode['steps/is_terminal'].values.numpy())),
                'steps/reward': tf.train.Feature(float_list=tf.train.FloatList(value=parsed_episode['steps/reward'].values)),
                'steps/observation/image': tf.train.Feature(bytes_list=tf.train.BytesList(value=parsed_episode['steps/observation/image'].values.numpy())),
                'steps/observation/joint_state': tf.train.Feature(float_list=tf.train.FloatList(value=parsed_episode['steps/observation/joint_state'].values)),
                'steps/observation/state': tf.train.Feature(float_list=tf.train.FloatList(value=parsed_episode['steps/observation/state'].values)),
                'steps/observation/wrist_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=parsed_episode['steps/observation/wrist_image'].values.numpy())),
            }

            example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            writer.write(example_proto.SerializeToString())

        except Exception as e:
            print(f"[Error] Episode {i} failed during low-level processing: {e}")
            traceback.print_exc()
            continue
            
    writer.close()
    print(f"✅ 全部处理完成，数据已保存至: {output_path}")



if __name__ == "__main__":
    names = ["libero_10_no_noops","libero_goal_no_noops","libero_object_no_noops","libero_spatial_no_noops"]
    data_dir = "/DATA/lmy/datasets/modified_libero_rlds"
    new_data_dir = "/DATA/lmy/datasets/modified_libero_rlds-250626"
    for name in names:
        #extract_instructions_to_json(dataset_name=name,data_dir=data_dir,)
        #accumulate_instructions(dataset_name=name)
        replace_instructions_and_save_low_level(dataset_name=name,data_dir=data_dir,split="train",new_dir=new_data_dir,new_name=name)