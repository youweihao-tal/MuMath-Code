

set -ex 

pwd=/path/to/MuMath-Code-src 

conda activate /path/to/env 

MODEL_SAVE=/path/to/saved_model 

cd $pwd 

for TEST_DATA_NAME in gsm8k math asdiv gsm-hard mawps svamp tabmwp 
do 

    SPLIT="test" 
    PROMPT_TYPE="mumathcode" 
    NUM_TEST_SAMPLE=-1 

    OUTPUT_DIR=/path/to/output_dir 
    OUTPUT_FILE=output_file_name 

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TOKENIZERS_PARALLELISM=true \
    python -um infer.inference \
    --model_name_or_path ${MODEL_SAVE} \
    --data_name ${TEST_DATA_NAME} \
    --data_dir $pwd/src/data \
    --split ${SPLIT} \
    --output_dir $OUTPUT_DIR \
    --output_file $OUTPUT_FILE \
    --prompt_type ${PROMPT_TYPE} \
    --use_train_prompt_format \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \

done 




