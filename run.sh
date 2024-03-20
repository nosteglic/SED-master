#!/bin/bash

export WANDB_API_KEY='6109ea69f151b0fa881f2c3a60db2ce11e9b8838'
export WANDB_MODE='offline'
export CUDA_VISIBLE_DEVICES=2
export WANDB_DIR=/data2/syx/results

timestamp=$(date '+%Y-%m-%d_%H:%M:%S')

cat << EOF
choose stage: [1, 2, 3]
stage 1: feature extraction
stage 2: train model
stage 3: test model
EOF
read stage

use_nohup=1
model_type=Conformer # CRNN/Conformer

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
  echo "no completed"
#  func_stage1_prompt(){
#    echo "--------------Stage 1: Extracting feature extraction--------------"
#    echo "config currently used: config/dcase_21_MT_$model_type.yaml"
#    echo "------------------------------------------------------------------"
#  }
#  func_stage1_prompt
#  fun_confirm_cfg "Please confirm your exp config again: [Y/N]" "Canceling..."
#  python local/feature_extraction.py --config config/dcase_21_MT_$model_type.yaml
#  # 检查退出状态
#  if [ $? -eq 0 ]; then
#    func_stage1_prompt
#    echo "Done"
#  fi
fi

if [ ${stage} -eq 2 ]; then
  func_stage2_prompt(){
    echo "--------------Stage 2: Training model--------------"
    echo "nohup? $use_nohup"
    echo "model_type: $model_type"
    echo "GPU ID: $CUDA_VISIBLE_DEVICES"
    echo "wandb mode: $WANDB_MODE"
    echo "*******idea config*******"
    cat config/idea.yaml
    echo -e "\n*************************"
  }
  func_stage2_prompt
  fun_confirm_cfg "Please confirm your exp config again: [Y/N]" "Canceling..."
  echo "Start training..."

  if [ $use_nohup -eq 1 ]; then
    nohup python -u src/methods/train_MT.py \
    --model_type $model_type \
    >> /data2/syx/results/logs/${model_type}_${timestamp}.log 2>&1 &
    pid=$!
    echo "PID: $pid"
  else
    python -u src/methods/train_MT.py \
    --model_type $model_type
    if [ $? -eq 0 ]; then
      func_stage2_prompt
      echo "Done"
    fi
  fi
fi

if [ ${stage} -eq 3 ]; then
  func_stage3_prompt(){
    echo "--------------Stage 3: Validing/Testing model--------------"
    echo "exp_name: $exp_name"
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
  read -p "Input your exp_name: " exp_name
  func_stage3_py
  if [ $revalid -eq 1 ]; then
    fun_confirm_cfg "Continue to test? [Y/N]" "Only revalid..."
    revalid=0
    func_stage3_py
  fi
fi
