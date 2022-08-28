export PYTORCH_PRETRAINED_BERT_CACHE=tmp/bert-cased-pretrained-cache
DATA_DIR="uav_dataset/uav-annonation/k_folk/"
EPOCH=$1
MODEL_RECOVER_PATH="tmp/new_finetuned_models/bert_save/model.${EPOCH}.bin"
EVAL_SPLIT=$2

python unilm/src/biunilm/decode_seq2seq.py --bert_model bert-base-cased --new_segment_ids --mode s2s --need_score_traces \
--input_file ${DATA_DIR}/uav.src.no.me.${EVAL_SPLIT}.txt --split ${EVAL_SPLIT} \
--rel_file ${DATA_DIR}/uav.src.me.5hop.${EVAL_SPLIT}.txt --split ${EVAL_SPLIT} \
--model_recover_path ${MODEL_RECOVER_PATH} \
--max_seq_length 128 --max_tgt_length 32 \
--batch_size 32 --beam_size 5 --length_penalty 0 \
--forbid_duplicate_ngrams --forbid_ignore_word "."


cp tmp/new_finetuned_models/bert_save/model.${EPOCH}.bin.${EVAL_SPLIT} decoded_sentences/${EVAL_SPLIT}