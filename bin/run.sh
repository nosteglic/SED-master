python local/feature_extraction.py --config config/dcase20.yaml --recompute True
python -u src/train.py --config config/dcase21.yaml
python -u src/test.py --exp_name ConformerSED_21 --test_meta /data2/syx/DCASE2021/metadata/eval/public.tsv --test_audio_dir /data2/syx/DCASE2021/audio/eval/public