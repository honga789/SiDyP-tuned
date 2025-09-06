"""
Script tạo dòng lệnh chạy main_tuned.py với các tham số cấu hình.
Chỉnh sửa giá trị các biến bên dưới cho phù hợp, sau đó chạy script để lấy dòng lệnh.
"""

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

def build_command(cfg):
	cmd = ["python src/main_tuned.py"]
	for k, v in cfg.items():
		if v is not None:
			if isinstance(v, str) and (" " in v or "," in v):
				cmd.append(f'--{k} "{v}"')
			else:
				cmd.append(f"--{k} {v}")
	return " ".join(cmd)

if __name__ == "__main__":
	print(build_command(config))
