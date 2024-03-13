#!/bin/bash

export WANDB_API_KEY='6109ea69f151b0fa881f2c3a60db2ce11e9b8838'
export WANDB_MODE='offline'
export CUDA_VISIBLE_DEVICES=3

stage=3
stop_stage=3

model_type=Conformer # CRNN/Conformer
batch_size=32
batch_size_sync=32
use_events=True
use_bg=False
use_sigmoid=True
use_mixup=False
beta=0.1
seed=7

exp_name="$model_type-b$batch_size-sync$batch_size_sync"

if [ "$use_events" = "True" ]; then
  exp_name="$exp_name-concat"
  if [ "$use_bg" = "True" ]; then
    exp_name="$exp_name-bg"
  fi
  if [ "$use_sigmoid" = "True" ]; then
    exp_name="$exp_name-sigmoid"
  else
    exp_name="$exp_name-nosigmoid"
  fi
  if [ "$use_mixup" = "True" ]; then
    exp_name="$exp_name-mixup"
  else
    exp_name="$exp_name-nomixup"
  fi
  exp_name="$exp_name-beta$beta"
fi
exp_name="$exp_name-seed$seed"
#exp_name="CRNN-b24-07"

on_test=True
revalid=True
test_mode=score # loss/score/psds

# stage还没有施工好
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "--------------Stage 1: Extracting feature extraction--------------"
  echo "config currently used: config/dcase_21_MT_$model_type.yaml"
  echo "------------------------------------------------------------------"
  python local/feature_extraction.py --config config/dcase_21_MT_$model_type.yaml
  echo "--------------Stage 1: Extracting feature extraction--------------"
  echo "config currently used: config/dcase_21_MT_$model_type.yaml"
  echo "Done"
  echo "------------------------------------------------------------------"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "--------------Stage 2: Training model--------------"
  echo "config currently used: config/dcase_21_MT_$model_type.yaml"
  echo "exp name: $exp_name"
  echo "model_type: $model_type"
  echo "batch_size: $batch_size"
  echo "batch_size_sync: $batch_size_sync"
  echo "use_events? $use_events"
  echo "use_bg? $use_bg"
  echo "use_sigmoid? $use_sigmoid"
  echo "use_mixup? $use_mixup"
  echo "beta: $beta"
  echo "seed: $seed"
  echo "---------------------------------------------------"

  nohup python -u src/methods/train_MT.py \
  --model_type $model_type \
  --batch_size $batch_size \
  --batch_size_sync $batch_size_sync \
  --use_events $use_events \
  --use_bg $use_bg \
  --use_sigmoid $use_sigmoid \
  --use_mixup $use_mixup \
  --beta $beta \
  --seed $seed \
  > logs/$exp_name.log 2>&1 &
  pid=$!
  echo "GPU ID: $CUDA_VISIBLE_DEVICES"
  echo "wandb mode: $WANDB_MODE"
  echo "PID: $pid"
  echo "---------------------------------------------------"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "--------------Stage 3: Validing/Testing model--------------"
  echo "model currently used: $exp_name"
  echo "on_test? $on_test"
  echo "revalid? $revalid"
  echo "mode: $test_mode"
  echo "------------------------------------------------------------"
  # 获取用户输入
  read -p "Please confirm your exp config again: [Y/N]" userInput
  # 判断用户输入
  if [ "$userInput" != "Y" ] && [ "$userInput" != "Yes" ] && [ "$userInput" != "y" ] && [ "$userInput" != "Yes" ]; then
      echo "Canceling..."
      exit 1  # 终止程序运行
  fi

  python -u src/methods/test_MT.py  \
  --exp_name $exp_name \
  --on_test $on_test \
  --revalid $revalid \
  --mode $test_mode

    # 检查退出状态
  if [ $? -eq 0 ]; then
      echo "--------------Stage 3: Validing/Testing model--------------"
      echo "model currently used: $exp_name"
      echo "on_test? $on_test"
      echo "revalid? $revalid"
      echo "mode: $test_mode"
      echo "Done"
      echo "------------------------------------------------------------"
  fi
  if [ "$revalid" = "False" ]; then
    read -p "Do you want to continue to test?: [Y/N]" userInput
    # 判断用户输入
    if [ "$userInput" != "Y" ] && [ "$userInput" != "Yes" ] && [ "$userInput" != "y" ] && [ "$userInput" != "Yes" ]; then
        echo "Only revalid..."
        exit 1  # 终止程序运行
    fi
    revalid=True
  fi
fi
