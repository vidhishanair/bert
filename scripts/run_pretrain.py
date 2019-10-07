SEQ_LEN=128
BERT_BASE_DIR=gs://fat_storage/pretrained_bert/wwm_uncased_L-24_H-1024_A-16
INPUT_DATA=gs://fat_storage/bert_pretraining_expts/data/nq_data/processed_data_mseq${SEQ_LEN}
OUTPUT=gs://fat_storage/bert_pretraining_expts/models/nq_models/output_test

python run_pretraining.py \
  --input_file=$INPUT_DATA/test.tfrecord \
  --output_dir=$OUTPUT \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=${SEQ_LEN} \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 \
  --use_tpu=True \
  --tpu_name=$TPU_NAME
