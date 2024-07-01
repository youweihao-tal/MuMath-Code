

set -ex 

MODEL_NAME_OR_PATH="gpt-4" 
DATA_NAME="math"

DATA_FILE=/path/to/question/file 
OUTPUT_DIR=/path/to/output_file 

SPLIT="train" 
PROMPT_TYPE="mumathcode" 
NUM_TEST_SAMPLE=-1 

python -um infer.request_solutions \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_file $DATA_FILE \
    --output_dir $OUTPUT_DIR \
    --data_name $DATA_NAME \
    --split $SPLIT \
    --prompt_type $PROMPT_TYPE \
    --num_test_sample $NUM_TEST_SAMPLE \
    --seed 0 \
    --n_sampling 10 \
    --temperature 0.8 \
    --top_p 1.0 \
    --start 0 \
    --end -1 \
    --num_threads 78 

