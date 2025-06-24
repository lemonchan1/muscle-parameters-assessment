from pathlib import Path

# 获取脚本所在目录
script_dir = Path(__file__).parent

# 定义子目录路径
data_path = script_dir / "data"
label_path = script_dir / "label"
result_path = script_dir / "result"

# 创建目录（如果不存在）
data_path.mkdir(parents=True, exist_ok=True)
label_path.mkdir(parents=True, exist_ok=True)
result_path.mkdir(parents=True, exist_ok=True)
