#!/usr/bin/env bash\

if [[ $HOSTNAME == *"ws802"* ]]
then
  # update the following paths accordingly
  DATA_HOME="/[data_home]/data"
  LOG_HOME="/[data_home]/outputs"
  PROJECT_DIR="/[project_home]/critical_wsl"
  MODEL_HOME="[data_home]/models"
else
  echo "[BASH CONFIGURATION] Unknown Host: [$HOSTNAME], User: [$USER]"
  exit 1
fi

CUDA_ID=0
STORE_MODEL=0
OUTPUT_PREFIX="tws_vanilla_save"
LOG_ROOT="${LOG_HOME}/remove_me/${OUTPUT_PREFIX}"
DATA_ROOT="${DATA_HOME}/wrench/tmp_data"
EXP_NAME="tws-vanilla-remove-me"

DATASET="agnews"

for j in 1234
do
for validation_label_seed in 1234 5678 42 24 6
do
for num_samples_per_class in 5 10 20 30 40 50
do
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$CUDA_ID python3 ${PROJECT_DIR}/main.py \
--dataset $DATASET \
--log_root $LOG_ROOT \
--data_root $DATA_ROOT \
--exp_name $EXP_NAME \
--trainer_name vanilla_small_validation \
--model_name roberta-base \
--num_samples_per_class $num_samples_per_class \
--nl_batch_size 32 \
--eval_batch_size 32 \
--max_sen_len 128 \
--lr 2e-5 \
--num_training_steps 10 \
--patience 30 \
--eval_freq 25 \
--train_eval_freq 25 \
--store_model $STORE_MODEL \
--validation_label_seed $validation_label_seed \
--manualSeed $j
done
done
done