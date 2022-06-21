export CUDA_VISIBLE_DEVICES=1
python main.py \
    --model_name_or_path bert-base-uncased \
    --data_path ./data \
    --output_dir ./results \
    --mode classify \
    --do_train True \
    --fp16 True \
    --epoch_num 5 \
    --batch_size 64 \
    --max_len 32 \
    --learning_rate 5e-5 \
    --clip_max_grad_norm 2 \
    --gradient_accumulation_steps 4 \
    "$@"
