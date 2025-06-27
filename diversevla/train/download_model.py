from huggingface_hub import snapshot_download

# 或者只下载特定子目录（如模型文件、权重等）
for i in range(20):
    try:
        snapshot_download(
            repo_id="TRI-ML/prismatic-vlms",
            repo_type="model",
            allow_patterns=["prism-dinosiglip-224px+7b/**"],
            local_dir="/share/lmy/models/prismatic-vlms",  # ← 你想保存的位置
            local_dir_use_symlinks=False  # 避免软链接，推荐设为 False
        )
        print("Successfully download prism-dinosiglip-224px+7b!")
        break

    except:
        continue