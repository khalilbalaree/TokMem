export CUDA_VISIBLE_DEVICES=0

python main_in_domain.py \
    --num_tasks 10 \
    --train_size 100 \
    --val_size 10 \
    --test_size 50 \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_length 1280 \
    --max_instruction_tokens 1024 \
    --eval_batch_size 32 \
    --validate_every_n_steps 100
    