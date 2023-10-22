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
EXP_NAME=tws-l2r
VAL_CLEAN=0

DATASET="agnews"

for j in 1234 5678 42 24 6
do
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$CUDA_ID python3 ${PROJECT_DIR}/main.py \
--dataset $DATASET \
--log_root $LOG_ROOT \
--data_root $DATA_ROOT \
--trainer_name l2r \
--model_name roberta-base \
--exp_name $EXP_NAME \
--validation_on_clean $VAL_CLEAN \
--nl_batch_size 16 \
--eval_batch_size 32 \
--max_sen_len 128 \
--lr 2e-5 \
--num_training_steps 6000 \
--patience 30 \
--eval_freq 25 \
--train_eval_freq 25 \
--store_model $STORE_MODEL \
--manualSeed $j
done