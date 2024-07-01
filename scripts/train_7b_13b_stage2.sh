


set -ex 

pwd=/path/to/MuMath-Code-src 

# MODEL_TYPE=llama 
# MODEL_TYPE=codellama 
MODEL_TYPE= 

MODEL_PATH=/path/to/stage1_trained_model 

if [ "$MODEL_TYPE" = "llama" ]; then 
    source activate /path/to/mumathcode_llama_env 
elif [ "$MODEL_TYPE" = "codellama" ]; then 
    source activate /path/to/mumathcode_codellama_env 
else 
    echo "wrong MODEL_TYPE" 
    exit 1 
fi 

TRAIN_FILE=/path/to/train_data_file 

MODEL_SAVE=/path/to/saved_model 

mkdir -p $MODEL_SAVE 

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

NUM_GPUS=8 

# ------------------- model -------------------
MODEL_SIZE=7b
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


accelerate launch \
    --main_process_port 18200 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file $DEEPSPEED \
    $pwd/train/finetune_stage2.py \
    --model_name_or_path ${MODEL_PATH} \
    --use_slow_tokenizer \
    --gradient_checkpointing \
    --train_file $TRAIN_FILE \
    --max_seq_length 2048 \
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
    --use_flash_attn \
    --mask_prompt 


