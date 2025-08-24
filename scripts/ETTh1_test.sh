model_name=LAMP

# testing the model on all forecast lengths
for test_pred_len in 96 192
do
python -u run_multidomain.py \
  --datasets ETTh2,ETTm2,ETTm1,electricity,traffic,weather \
  --config_path ./configs/multiple_datasets.yml \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --electri_multiplier 1 \
  --traffic_multiplier 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1 \
  --llm_ckp_dir gpt2\
  --model $model_name \
  --data ETTh1 \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len $test_pred_len \
  --batch_size 256 \
  --learning_rate 0.001 \
  --patience 3 \
  --mlp_hidden_dim 2048 \
  --train_epochs 10 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --target_data ETTh1 \
  --test_data_path ETTh1.csv \
  --loss MAE 
done