#!/bin/bash

export WANDB_API_KEY='6109ea69f151b0fa881f2c3a60db2ce11e9b8838'
export WANDB_MODE='offline'
export CUDA_VISIBLE_DEVICES=3

stage=2
use_nohup=1
debugmode=1
model_type=Conformer # CRNN/Conformer
batch_size=32
batch_size_sync=32
use_clean=0
use_events=1
use_mixup=1
use_bg=0
use_sigmoid=-1
beta=-1
seed=7

exp_name="$model_type-b$batch_size-sync$batch_size_sync"

if [ $use_clean -eq 1 ]; then
  exp_name="$exp-clean"
fi
if [ $use_events -eq 1 ]; then
  exp_name="$exp_name-concat"
  if [ $use_bg -eq 1 ]; then
    exp_name="$exp_name-bg"
  fi
  if [ $use_sigmoid -eq 1 ]; then
    exp_name="$exp_name-sigmoid"
  elif [ $use_sigmoid -eq 0 ]; then
    exp_name="$exp_name-nosigmoid"
  fi
  if [ $use_mixup -eq 1 ]; then
    exp_name="$exp_name-mixup"
  else
    exp_name="$exp_name-nomixup"
  fi
  if [ $beta != "-1" ]; then
    exp_name="$exp_name-beta$beta"
  fi

fi
exp_name="$exp_name-seed$seed"
#exp_name="Conformer-b32-sync32-concat-nosigmoid-mixup-beta-1.0-seed7"

on_test=1
revalid=1
test_mode=score # loss/score/psds

fun_confirm_cfg(){
  # 获取用户输入
    read -p "$1" userInput
    # 判断用户输入
    if [ "$userInput" != "Y" ] && [ "$userInput" != "Yes" ] && [ "$userInput" != "y" ] && [ "$userInput" != "Yes" ]; then
        echo "$2"
        exit 1  # 终止程序运行
    fi
}

# stage还没有施工好
if [ ${stage} -eq 1 ]; then
  func_stage1_prompt(){
    echo "--------------Stage 1: Extracting feature extraction--------------"
    echo "config currently used: config/dcase_21_MT_$model_type.yaml"
    echo "------------------------------------------------------------------"
  }
  func_stage1_prompt
  fun_confirm_cfg "Please confirm your exp config again: [Y/N]" "Canceling..."
  python local/feature_extraction.py --config config/dcase_21_MT_$model_type.yaml
  # 检查退出状态
  if [ $? -eq 0 ]; then
    func_stage1_prompt
    echo "Done"
  fi
fi

if [ ${stage} -eq 2 ]; then
  func_stage2_prompt(){
    echo "--------------Stage 2: Training model--------------"
    echo "config currently used: config/dcase_21_MT_$model_type.yaml"
    echo "nohup? $use_nohup"
    echo "exp name: $exp_name"
    echo "model_type: $model_type"
    echo "batch_size: $batch_size"
    echo "batch_size_sync: $batch_size_sync"
    echo "use_clean? $use_clean"
    echo "use_events? $use_events"
    echo "use_bg? $use_bg"
    echo "use_sigmoid? $use_sigmoid"
    echo "use_mixup? $use_mixup"
    echo "beta: $beta"
    echo "seed: $seed"
    echo "GPU ID: $CUDA_VISIBLE_DEVICES"
    echo "wandb mode: $WANDB_MODE"
    echo "---------------------------------------------------"
  }
  func_stage2_prompt
  fun_confirm_cfg "Please confirm your exp config again: [Y/N]" "Canceling..."
  if [ $use_nohup -eq 1 ]; then
    nohup python -u src/methods/train_MT.py \
    --exp_name $exp_name \
    --model_type $model_type \
    --batch_size $batch_size \
    --batch_size_sync $batch_size_sync \
    --use_events $use_events \
    --use_bg $use_bg \
    --use_sigmoid $use_sigmoid \
    --use_mixup $use_mixup \
    --beta $beta \
    --use_clean $use_clean \
    --seed $seed \
    --debugmode $debugmode \
    > logs/$exp_name.log 2>&1 &
  else
    python -u src/methods/train_MT.py \
    --model_type $model_type \
    --batch_size $batch_size \
    --batch_size_sync $batch_size_sync \
    --use_events $use_events \
    --use_bg $use_bg \
    --use_sigmoid $use_sigmoid \
    --use_mixup $use_mixup \
    --beta $beta \
    --seed $seed \
    --debugmode $debugmode
  fi
  pid=$!
  echo "PID: $pid"
  echo "---------------------------------------------------"
fi

if [ ${stage} -eq 3 ]; then
  func_stage3_prompt(){
    echo "--------------Stage 3: Validing/Testing model--------------"
    echo "model currently used: $exp_name"
    echo "on_test? $on_test"
    echo "revalid? $revalid"
    echo "mode: $test_mode"
    echo "------------------------------------------------------------"
  }
  func_stage3_py() {
    func_stage3_prompt
    fun_confirm_cfg "Please confirm your exp config again: [Y/N]" "Canceling..."
    python -u src/methods/test_MT.py  \
    --exp_name $exp_name \
    --on_test $on_test \
    --revalid $revalid \
    --mode $test_mode
     # 检查退出状态
    if [ $? -eq 0 ]; then
        func_stage3_prompt
        echo "Done"
        echo "------------------------------------------------------------"
    else
      echo "model currently used: $exp_name is cancelled"
      revalid=0
    fi
  }
  func_stage3_py
  if [ $revalid -eq 1 ]; then
    fun_confirm_cfg "Continue to test? [Y/N]" "Only revalid..."
    revalid=0
    func_stage3_py
  fi
fi
