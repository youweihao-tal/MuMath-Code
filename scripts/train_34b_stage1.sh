
set -ex 

pwd=/path/to/MuMath-Code-src 

MODEL_TYPE=codellama 

conda activate /path/to/env 

MODEL_PATH=/path/to/CodeLlama-34b-Python-hf 

TRAIN_FILE=/path/to/train_data_file 

MODEL_SAVE=/path/to/saved_model 

mkdir -p $MODEL_SAVE 

NUM_GPUS=16 

# ------------------- model -------------------
MODEL_SIZE=34b
if [ "$MODEL_SIZE" = "7b" ] || [ "$MODEL_SIZE" = "13b" ]; then
    LEARNING_RATE=2e-5
    DEEPSPEED=$pwd/ds_configs/stage3_no_offload_accelerate.conf
    BATCH_SIZE_PER_GPU=16 
elif [ "$MODEL_SIZE" = "34b" ]; then 
    LEARNING_RATE=1e-5
    DEEPSPEED=$pwd/ds_configs/stage3_offload_optim_accelerate.conf 
    BATCH_SIZE_PER_GPU=8 
else
    echo "MODEL_SIZE should be 7b, 13b, 34b"
    exit 1
fi


NUM_TRAIN_EPOCHS=3 
TOTAL_BATCH_SIZE=128 
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"


if [[ $IS_MASTER == 1 ]];then 
  DISTRIBUTED_ARGS="--num_processes $NUM_GPUS --num_machines $WORLD_SIZE --machine_rank $RANK --main_process_ip $HOSTNAME --main_process_port $MASTER_PORT"
else
  DISTRIBUTED_ARGS="--num_processes $NUM_GPUS --num_machines $WORLD_SIZE --machine_rank $RANK --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT"
fi


accelerate launch \
    $DISTRIBUTED_ARGS \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file $DEEPSPEED \
    $pwd/train/finetune_stage1.py \
    --model_name_or_path ${MODEL_PATH} \
    --use_slow_tokenizer \
    --gradient_checkpointing \
    --train_file $TRAIN_FILE \
    --max_seq_length 1536 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --output_dir $MODEL_SAVE \
    --logging_steps 1 \
    --mask_prompt \
    --checkpointing_steps 10000 



