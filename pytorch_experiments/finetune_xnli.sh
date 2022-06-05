#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --partition gpu
#SBATCH --time 32:00:00
#SBATCH --mem 32000
#SBATCH --gres=gpu:tesla:4
conda activate IWSLT_wav2vec_bart
# python create_model.py
# --model_name_or_path microsoft/infoxlm-base \
CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/run_xnli.py \
  --model_name_or_path ./xnli_ar_en/checkpoint-30680 \
  --language ur \
  --train_language en \
  --do_eval \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-6 \
  --num_train_epochs 10.0 \
  --max_seq_length 64 \
  --output_dir xnli_ar_en \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --logging_strategy epoch \
  --load_best_model_at_end True \
  --weight_decay=0.01 \
  --warmup_steps=5000
