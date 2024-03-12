#!/bin/bash

stage=3
stop_stage=3

model_type=Conformer # CRNN/Conformer
model_cfg=b32-07
config_yaml="config/dcase21_MT_$model_type.yaml"
exp_name="$model_type-$model_cfg"
on_test=True
revalid=False
test_mode=score # loss/score/psds

# stage还没有施工好
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "--------------Stage 1: Extracting feature extraction--------------"
  echo "config currently used: $config_yaml"
  echo "------------------------------------------------------------------"
  python local/feature_extraction.py --config $config_yaml
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "--------------Stage 2: Training model--------------"
  echo "config currently used: $config_yaml"
  echo "---------------------------------------------------"

  if [ "$model_type" = "Conformer" ]; then
    python -u src/methods/train_MT.py --config $config_yaml
  elif [ "$model_type" = "CRNN" ]; then
    python -u src/methods/train_MT_CRNN.py --config $config_yaml
  else
    echo "error: no such model"
  fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "--------------Stage 3: Validing/Testing model--------------"
  echo "model currently used: $exp_name"
  echo "on_test? $on_test"
  echo "revalid? $revalid"
  echo "mode: $test_mode"
  echo "------------------------------------------------------------"
  if [ "$model_type" = "Conformer" ]; then
    python -u src/methods/test_MT.py  \
    --exp_name $exp_name \
    --on_test $on_test \
    --revalid $revalid \
    --mode $test_mode
  elif [ "$model_type" = "CRNN" ]; then
    python -u src/methods/test_MT_CRNN.py \
    --exp_name $exp_name \
    --on_test $on_test \
    --revalid $revalid \
    --mode $test_mode
  else
    echo "error: no such model"
  fi
fi
