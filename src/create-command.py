"""
Script tạo dòng lệnh chạy main_tuned.py với các tham số cấu hình.
Tên file log = dataset + '_' + tên_cuối_cùng_của_train_feather_path (bỏ phần mở rộng)
+ '_' + timestamp. Ví dụ: logs/agnews_agnews-12k-train_LLM_20250906_191530.log
"""

import os
import re

config = {
    "seed": 42,
    "dataset": "fashion-mnist",
    "noise_type": "llm",
    "data_type": "image",

    "train_csv_path": "datasets/Fashion-MNIST-test/fashion_mnist.csv",
    "train_image_path": "datasets/Fashion-MNIST-test/images",
    "train_feather_path": "datasets/Fashion-MNIST-test/fashion-mnist-test-clip-b16-noise/fashion-mnist-test_LLM.feather",
    "train_data_column": "image_name",
    "train_label_column": "label",

    "test_csv_path": "datasets/fashion-mnist-2k5-testset/fashion-mnist-test-2k5.csv",
    "test_image_path": "datasets/fashion-mnist-2k5-testset/images",
    "test_data_column": "image_name",
    "test_label_column": "label",

    "num_classes": 10,

    "plc": "clip",
    "embed": "clip",

    "num_workers": 4,
    "train_batch_size": 32,
    "eval_batch_size": 32,
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
