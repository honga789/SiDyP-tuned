
python src/main.py --dataset semeval --noise_type llm --llm_type llama3-70b --plc_epochs 20 --noise_ratio 0.5 --clip_sample True \
    --train_batch_size 128 --eval_batch_size 128 --seed 1 --prompt_type zeroshot --diff_epochs 10 --warmup_epochs 0.2 \
    --diff_lr 6e-4 --K 10 --certain_threshold 0.9 --dominant_threshold 0.8 --num_sample 4 --train_timesteps 800 --infer_timesteps 10 \
