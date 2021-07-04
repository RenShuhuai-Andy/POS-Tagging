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
BATCH_SIZE=32
SEED=888
DATA_DIR=data/final_eval/$DATA_TYPE
#BERT_PATH=results/bert-wwm-ext/$DATA_TYPE-crf
MAX_LENGTH=128
LOSS_TYPE=CrossEntropyLoss

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 token-classification/run_pos_crf.py \
   --data_dir $DATA_DIR \
   --labels $DATA_DIR/labels.txt \
   --model_name_or_path $BERT_PATH \
   --output_dir $BERT_PATH \
   --max_seq_length  $MAX_LENGTH \
   --per_device_eval_batch_size $BATCH_SIZE \
   --seed $SEED \
   --loss_type $LOSS_TYPE \
   --do_predict