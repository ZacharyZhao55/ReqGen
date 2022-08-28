DATA_DIR=uav_dataset/; \
OUTPUT_DIR=tmp/new_finetuned_models/; \
MODEL_RECOVER_PATH=tmp/unilmv1-large-cased-uav.bin; \
BERT_MODEL = /media/g311/DATA1/za/commgen_unilm/tmp/bert-large-cased.tar.gz;\
export PYTORCH_PRETRAINED_BERT_CACHE=tmp/bert-cased-pretrained-cache; \
python unilm/src/biunilm/run_seq2seq.py --do_train --num_workers 0 \
  --bert_model bert-large-cased --new_segment_ids \
  --data_dir ${DATA_DIR} \
  --src_file uav.src.train.10.txt \
  --tgt_file uav.tgt.train.10.txt \
  --rel_file uav.rel.train.10.txt \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 64 --max_position_embeddings 64 \
  --always_truncate_tail  --max_len_a 64 --max_len_b 64 \
  --mask_prob 0.7 --max_pred 20 \
  --train_batch_size 32 --gradient_accumulation_steps 4 \
  --learning_rate 0.00001 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 30 \
