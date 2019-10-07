SEQ_LEN=128

DATA_DIR=gs://fat_storage/bert_pretraining_expts/data/nq_data/raw_data
OUTPUT=gs://fat_storage/bert_pretraining_expts/data/nq_data/processed_data_mseq${SEQ_LEN}
NQ_BASE=gs://fat_storage/bert-joint-baseline

python create_pretraining_data.py \
  --input_file=$DATA_DIR/*.txt \
  --output_file=$OUTPUT/nq-train-00.tfrecord,$OUTPUT/nq-train-01.tfrecord,$OUTPUT/nq-train-02.tfrecord,$OUTPUT/nq-train-03.tfrecord \
  --vocab_file=$NQ_BASE/vocab-nq.txt \
  --do_lower_case=True \
  --do_whole_word_mask=True \
  --max_seq_length=${SEQ_LEN} \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
