"""
Script tạo dòng lệnh chạy main_tuned.py với các tham số cấu hình.
Tên file log = dataset + '_' + tên_cuối_cùng_của_train_feather_path (bỏ phần mở rộng)
+ '_' + timestamp. Ví dụ: logs/agnews_agnews-12k-train_LLM_20250906_191530.log
"""

import os
import re

config = {
    "seed": 42,
    "dataset": "agnews",
    "train_csv_path": "datasets/Agnews-12k-train/ag_news_12k.csv",
    "train_feather_path": "datasets/Agnews-12k-train/agnews-12k-train-bert-noise/agnews-12k-train_LLM.feather",
    "train_data_column": "text",
    "train_label_column": "label",
    "test_csv_path": "datasets/agnews-3k-testset/agnews-test-3k.csv",
    "test_data_column": "text",
    "test_label_column": "label",
    "num_classes": 4,
    "embed": "bert-base-uncased",
    "train_batch_size": 64,
    "eval_batch_size": 64,
}

def _sanitize(name: str) -> str:
    # Chỉ giữ chữ/số/dấu chấm/gạch/underscore để an toàn khi đặt tên file
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name.strip())

def _log_file_from_cfg(cfg) -> str:
    dataset = _sanitize(str(cfg.get("dataset", "run")))
    feather = str(cfg.get("train_feather_path", "")).strip()
    if feather:
        base = os.path.basename(feather)                 # agnews-12k-train_LLM.feather
        stem, _ = os.path.splitext(base)                 # agnews-12k-train_LLM
    else:
        stem = "unknown"
    stem = _sanitize(stem)
    # tên + timestamp
    return f'logs/{dataset}_{stem}_$(date +%Y%m%d_%H%M%S).log'

def build_command(cfg):
    cmd = ["python src/main_tuned.py"]
    for k, v in cfg.items():
        if v is None:
            continue
        if isinstance(v, str) and (" " in v or "," in v):
            cmd.append(f'--{k} "{v}"')
        else:
            cmd.append(f"--{k} {v}")
    log_file = _log_file_from_cfg(cfg)
    cmd.append(f'2>&1 | tee "{log_file}"')
    return " ".join(cmd)

if __name__ == "__main__":
    # Nhớ tạo thư mục logs trước khi chạy:
    #   mkdir -p logs
    print("mkdir -p logs")
    print(build_command(config))
