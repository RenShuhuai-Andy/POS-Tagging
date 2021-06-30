while getopts ":b:d:c:" opt
do
    case $opt in
        b)
          echo "BERT_PATH=$OPTARG"
          BERT_PATH=$OPTARG
        ;;
        d)
          echo "DATA_TYPE=$OPTARG"
          DATA_TYPE=$OPTARG
        ;;
        c)
          echo "CUDA_VISIBLE_DEVICES=$OPTARG"
          CUDA_VISIBLE_DEVICES=$OPTARG
        ;;
        ?)
          echo "unknown parameters"
        exit 1;;
    esac
done

#DATA_TYPE=simplified
OUTPUT_DIR=results/final_eval/$BERT_PATH/$DATA_TYPE
BATCH_SIZE=32
NUM_EPOCHS=10
SAVE_STEPS=250
SEED=888
DATA_DIR=data/final_eval/$DATA_TYPE
#BERT_PATH=bert-wwm-ext
MAX_LENGTH=128
LEARNING_RATE=3e-5
LOSS_TYPE=CrossEntropyLoss
LOSS_GAMMA=2
LOGGING_STEPS=50

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 token-classification/run_pos.py \
   --data_dir $DATA_DIR --learning_rate $LEARNING_RATE \
   --labels $DATA_DIR/labels.txt \
   --model_name_or_path $BERT_PATH \
   --output_dir $OUTPUT_DIR \
   --max_seq_length  $MAX_LENGTH \
   --num_train_epochs $NUM_EPOCHS \
   --per_device_train_batch_size $BATCH_SIZE \
   --per_device_eval_batch_size 32 \
   --save_steps $SAVE_STEPS \
   --eval_steps $SAVE_STEPS \
   --evaluation_strategy "steps" \
   --seed $SEED \
   --loss_type $LOSS_TYPE \
   --loss_gamma $LOSS_GAMMA \
   --do_train \
   --do_eval \
   --logging_steps $LOGGING_STEPS \
   --load_best_model_at_end \
   --metric_for_best_model f1 \
   --logging_dir $OUTPUT_DIR"/runs/"