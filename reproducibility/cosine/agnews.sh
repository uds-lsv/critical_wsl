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
OUTPUT_PREFIX="tws_cosine_save"
LOG_ROOT="${LOG_HOME}/remove_me/${OUTPUT_PREFIX}"
DATA_ROOT="${DATA_HOME}/wrench/tmp_data"
EXP_NAME="tws-vanilla-remove-me"

# To apply the COSINE method, one must first train a teacher model using the vanilla method.

DATASET="agnews"
MODEL_SUFFIX="vanilla_noisy_validation/${DATASET}"
for T2 in 6000
do
for T3 in 50 100 200 300
do
for self_training_contrastive_weight in 1.0
do
for cosine_distmetric in 'cos'
do
for self_training_eps in 0.4 0.5 0.6 0.7 0.8
do
for self_training_confreg in 0.05 0.1
do
for j in 1234 5678
do
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$CUDA_ID python3 ${PROJECT_DIR}/main.py \
--dataset $DATASET \
--log_root $LOG_ROOT \
--data_root $DATA_ROOT \
--validation_on_clean 1 \
--trainer_name cosine \
--model_name roberta-base \
--exp_name $EXP_NAME \
--nl_batch_size 2 \
--eval_batch_size 2 \
--max_sen_len 32 \
--lr 2e-5 \
--T2 $T2 \
--T3 $T3 \
--cosine_distmetric $cosine_distmetric \
--self_training_eps $self_training_eps \
--self_training_confreg $self_training_confreg \
--self_training_contrastive_weight $self_training_contrastive_weight \
--teacher_init_weights_dir $MODEL_HOME/tws/$MODEL_SUFFIX \
--patience 100 \
--eval_freq 20 \
--train_eval_freq 20 \
--store_model $STORE_MODEL \
--manualSeed $j
done
done
done
done
done
done
done