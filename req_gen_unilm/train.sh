DATA_DIR=uav_dataset/uav-annonation/k_folk/; \
OUTPUT_DIR=tmp/new_finetuned_models/; \
MODEL_RECOVER_PATH=tmp/unilmv1-base-cased-uav.bin; \
BERT_MODEL = tmp/bert-base-cased.tar.gz;\
export PYTORCH_PRETRAINED_BERT_CACHE=tmp/bert-cased-pretrained-cache; \
CUDA_VISIBLE_DEVICES=0,1,2,3 python unilm/src/biunilm/run_seq2seq.py --do_train --num_workers 0 \
  --bert_model bert-base-cased --new_segment_ids \
  --data_dir ${DATA_DIR} \
  --src_file uav.src.train.txt \
  --tgt_file uav.tgt.train.txt \
  --rel_file uav.src.me.5hop.train.txt \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 128 --max_position_embeddings 128 \
  --always_truncate_tail  --max_len_a 64 --max_len_b 64 \
  --mask_prob 0.7 --max_pred 20 \
  --train_batch_size 32 --gradient_accumulation_steps 2 \
  --learning_rate 0.00005 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 30 \
