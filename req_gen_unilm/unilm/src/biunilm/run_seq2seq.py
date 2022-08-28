"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import glob
import math
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm, trange
import numpy as np
import re
from random import random as rand

import torch
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTrainingLossMask, BertModel
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from nn.data_parallel import DataParallelImbalance
import biunilm.seq2seq_loader as seq2seq_loader
import torch.distributed as dist

from knowledge_injection.knowledge_injection import knowledge_injection
from transformers import BertTokenizer as TransBertTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(Path(fn).stem.split('.')[-1]) for fn in fn_model_list]
                   ) & set([int(Path(fn).stem.split('.')[-1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None

def filter_n_hop_rel(sen):
    rel_list = sen.strip().split('.')
    filter_result = []
    for rel in rel_list:
        # print(rel)
        pattern = '#(.+?)#'
        hop = re.findall(pattern, rel)
        if len(hop) > 0:
            hop_value = int(hop[0])
        else:
            hop_value = 1
        if hop_value == 1:
            filter_result.append(rel)
        elif hop_value == 2 and rand() < 0.5:
            filter_result.append(rel)
        elif hop_value == 3 and rand() < 0.25:
            filter_result.append(rel)
        elif hop_value == 4 and rand() < 0.1:
            filter_result.append(rel)
        elif hop_value == 5 and rand() < 0.05:
            filter_result.append(rel)
    filter_sen = ' '.join(filter_result)
    return filter_sen



def relation_tokenizer(concept_rel, max_len, tokenizer, bert_tokenizer, bert_model):
    extended_attention_mask = None
    input_ids = None
    for sen in concept_rel:
        sen = filter_n_hop_rel(sen)
        pattern = '#(.+?)#'
        sen_no_hop_label = re.sub(pattern, '', sen)
        sen_token = tokenizer.tokenize(text=sen_no_hop_label)

        rel_input_mask = torch.zeros(max_len, max_len, dtype=torch.long)
        rel_input_mask[:, :len(sen_token) + 2].fill_(1)

        if input_ids == None:
            input_ids = bert_tokenizer.encode(
                sen_no_hop_label,
                add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
                max_length=max_len,  # 设定最大文本长度
                pad_to_max_length=True,  # pad到最大的长度
                return_tensors='pt',  # 返回的类型为pytorch tensor
                truncation=True
            )
            extended_attention_mask = bert_model.get_extended_attention_mask(input_ids, token_type_ids=None,
                                                                            attention_mask=rel_input_mask).permute(2, 1,0,3)
        else:
            new_input_ids = bert_tokenizer.encode(
                sen_no_hop_label,
                add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
                max_length=max_len,  # 设定最大文本长度
                pad_to_max_length=True,  # pad到最大的长度
                return_tensors='pt',  # 返回的类型为pytorch tensor
                truncation=True
            )
            input_ids = torch.cat((input_ids, new_input_ids), dim=0)
            new_extended_attention_mask = bert_model.get_extended_attention_mask(new_input_ids, token_type_ids=None,
                                                                                attention_mask=rel_input_mask).permute(2, 1, 0, 3)
            extended_attention_mask = torch.cat((extended_attention_mask, new_extended_attention_mask), dim=0)

    input_ids = input_ids
    inp_ebd = bert_model.embeddings(input_ids, token_type_ids=None, task_idx=None)
    return inp_ebd, extended_attention_mask


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--src_file", default=None, type=str,
                        help="The input data file name.")
    parser.add_argument("--tgt_file", default=None, type=str,
                        help="The output data file name.")
    parser.add_argument("--rel_file", default=None, type=str,
                        help="The rel data file name.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Bert config file path.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_dir",
                        default='',
                        type=str,
                        required=True,
                        help="The output directory where the log will be written.")
    parser.add_argument("--model_recover_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--optim_recover_path",
                        default=None,
                        type=str,
                        help="The file of pretraining optimizer.")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--label_smoothing", default=0, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--finetune_decay",
                        action='store_true',
                        help="Weight decay to the original weights.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
                        help="Dropout rate for hidden states.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float,
                        help="Dropout rate for attention probabilities.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp32_embedding', action='store_true',
                        help="Whether to use 32-bit float precision instead of 16-bit for embeddings")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument('--from_scratch', action='store_true',
                        help="Initialize parameters with random values (i.e., training from scratch).")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--new_pos_ids', action='store_true',
                        help="Use new position ids for LMs.")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--max_len_a', type=int, default=0,
                        help="Truncate_config: maximum length of segment A.")
    parser.add_argument('--max_len_b', type=int, default=0,
                        help="Truncate_config: maximum length of segment B.")
    parser.add_argument('--trunc_seg', default='',
                        help="Truncate_config: first truncate segment A/B (option: a, b).")
    parser.add_argument('--always_truncate_tail', action='store_true',
                        help="Truncate_config: Whether we should always truncate tail.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument("--mask_prob_eos", default=0, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument('--max_pred', type=int, default=20,
                        help="Max tokens of prediction.")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="Number of workers for the data loader.")

    parser.add_argument('--mask_source_words', action='store_true',
                        help="Whether to mask source words for training")
    parser.add_argument('--skipgram_prb', type=float, default=0.0,
                        help='prob of ngram mask')
    parser.add_argument('--skipgram_size', type=int, default=1,
                        help='the max size of ngram mask')
    parser.add_argument('--mask_whole_word', action='store_true',
                        help="Whether masking a whole word.")
    parser.add_argument('--do_l2r_training', action='store_true',
                        help="Whether to do left to right training")
    parser.add_argument('--has_sentence_oracle', action='store_true',
                        help="Whether to have sentence level oracle for training. "
                             "Only useful for summary generation")
    parser.add_argument('--max_position_embeddings', type=int, default=None,
                        help="max position embeddings")
    parser.add_argument('--relax_projection', action='store_true',
                        help="Use different projection layers for tasks.")
    parser.add_argument('--ffn_type', default=0, type=int,
                        help="0: default mlp; 1: W((Wx+b) elem_prod x);")
    parser.add_argument('--num_qkv', default=0, type=int,
                        help="Number of different <Q,K,V>.")
    parser.add_argument('--seg_emb', action='store_true',
                        help="Using segment embedding for self-attention.")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")

    args = parser.parse_args()

    assert Path(args.model_recover_path).exists(
    ), "--model_recover_path doesn't exist"

    args.output_dir = args.output_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))
    args.log_dir = args.log_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        dist.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    if args.local_rank not in (-1, 0):
        # Make sure only the first process in distributed training will download model & vocab
        dist.barrier()
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)
    if args.max_position_embeddings:
        tokenizer.max_len = args.max_position_embeddings
    data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer
    if args.local_rank == 0:
        dist.barrier()

    kj = knowledge_injection(model_name=r"/root/Desktop/uav_req_gen_unilm_v2/tmp",
                                 inputs_dim=768, num_hiddens=384, enc_hiddens_layers=4, att_hiddens_layers=8,
                                 dropout=0.1, output_dim=768)

    if args.do_train:
        print("Loading Train Dataset", args.data_dir)
        bi_uni_pipeline = [seq2seq_loader.Preprocess4Seq2seq(args.max_pred, args.mask_prob, list(tokenizer.vocab.keys(
        )), tokenizer.convert_tokens_to_ids, args.max_seq_length, new_segment_ids=args.new_segment_ids,
                                                             truncate_config={'max_len_a': args.max_len_a,
                                                                              'max_len_b': args.max_len_b,
                                                                              'trunc_seg': args.trunc_seg,
                                                                              'always_truncate_tail': args.always_truncate_tail},
                                                             mask_source_words=args.mask_source_words,
                                                             skipgram_prb=args.skipgram_prb,
                                                             skipgram_size=args.skipgram_size,
                                                             mask_whole_word=args.mask_whole_word, mode="s2s",
                                                             has_oracle=args.has_sentence_oracle, num_qkv=args.num_qkv,
                                                             s2s_special_token=args.s2s_special_token,
                                                             s2s_add_segment=args.s2s_add_segment,
                                                             s2s_share_segment=args.s2s_share_segment,
                                                             pos_shift=args.pos_shift)]
        file_oracle = None
        if args.has_sentence_oracle:
            file_oracle = os.path.join(args.data_dir, 'train.oracle')
        fn_src = os.path.join(
            args.data_dir, args.src_file if args.src_file else 'train.src')
        fn_tgt = os.path.join(
            args.data_dir, args.tgt_file if args.tgt_file else 'train.tgt')
        fn_rel = os.path.join(
            args.data_dir, args.rel_file if args.rel_file else 'train.rel')
        train_dataset = seq2seq_loader.Seq2SeqDataset(
            fn_src, fn_tgt, fn_rel, args.train_batch_size, data_tokenizer, args.max_seq_length, file_oracle=file_oracle,
            bi_uni_pipeline=bi_uni_pipeline)
        rel_list = []
        with open(fn_rel, "r", encoding='utf-8') as f_rel:
            for rel in f_rel:
                rel_list.append(rel.strip())
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset, replacement=False)
            _batch_size = args.train_batch_size
        else:
            train_sampler = DistributedSampler(train_dataset)
            _batch_size = args.train_batch_size // dist.get_world_size()
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, sampler=train_sampler,
                                                       num_workers=args.num_workers,
                                                       collate_fn=seq2seq_loader.batch_list_to_batch_tensors,
                                                       pin_memory=False)

    # note: args.train_batch_size has been changed to (/= args.gradient_accumulation_steps)
    # t_total = int(math.ceil(len(train_dataset.ex_list) / args.train_batch_size)
    t_total = int(len(train_dataloader) * args.num_train_epochs /
                  args.gradient_accumulation_steps)

    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # Prepare model
    recover_step = _get_max_epoch_model(args.output_dir)
    cls_num_labels = 2
    type_vocab_size = 6 + \
                      (1 if args.s2s_add_segment else 0) if args.new_segment_ids else 2
    num_sentlvl_labels = 2 if args.has_sentence_oracle else 0
    relax_projection = 4 if args.relax_projection else 0
    if args.local_rank not in (-1, 0):
        # Make sure only the first process in distributed training will download model & vocab
        dist.barrier()
    if (recover_step is None) and (args.model_recover_path is None):
        # if _state_dict == {}, the parameters are randomly initialized
        # if _state_dict == None, the parameters are initialized with bert-init
        _state_dict = {} if args.from_scratch else None
        model = BertForPreTrainingLossMask.from_pretrained(
            args.bert_model, state_dict=_state_dict, num_labels=cls_num_labels, num_rel=0,
            type_vocab_size=type_vocab_size, config_path=args.config_path, task_idx=3,
            num_sentlvl_labels=num_sentlvl_labels, max_position_embeddings=args.max_position_embeddings,
            label_smoothing=args.label_smoothing, fp32_embedding=args.fp32_embedding, relax_projection=relax_projection,
            new_pos_ids=args.new_pos_ids, ffn_type=args.ffn_type, hidden_dropout_prob=args.hidden_dropout_prob,
            attention_probs_dropout_prob=args.attention_probs_dropout_prob, num_qkv=args.num_qkv, seg_emb=args.seg_emb)
        global_step = 0
    else:
        if recover_step:
            logger.info("***** Recover model: %d *****", recover_step)
            model_recover = torch.load(os.path.join(
                args.output_dir, "model.{0}.bin".format(recover_step)), map_location='cpu')
            # recover_step == number of epochs
            global_step = math.floor(
                recover_step * t_total / args.num_train_epochs)
        elif args.model_recover_path:
            logger.info("***** Recover model: %s *****",
                        args.model_recover_path)
            model_recover = torch.load(
                args.model_recover_path, map_location='cpu')
            global_step = 0
        model = BertForPreTrainingLossMask.from_pretrained(
            '/root/Desktop/uav_req_gen_unilm_v2/tmp/bert-base-cased.tar.gz', state_dict=model_recover,
            num_labels=cls_num_labels, num_rel=0, type_vocab_size=type_vocab_size, config_path=args.config_path,
            task_idx=3, num_sentlvl_labels=num_sentlvl_labels, max_position_embeddings=args.max_position_embeddings,
            label_smoothing=args.label_smoothing, fp32_embedding=args.fp32_embedding, relax_projection=relax_projection,
            new_pos_ids=args.new_pos_ids, ffn_type=args.ffn_type, hidden_dropout_prob=args.hidden_dropout_prob,
            attention_probs_dropout_prob=args.attention_probs_dropout_prob, num_qkv=args.num_qkv, seg_emb=args.seg_emb)
    if args.local_rank == 0:
        dist.barrier()

    if args.fp16:
        model.half()
        if args.fp32_embedding:
            model.bert.embeddings.word_embeddings.float()
            model.bert.embeddings.position_embeddings.float()
            model.bert.embeddings.token_type_embeddings.float()
    model.to(device)
    # 分佈式訓練
    if args.local_rank != -1:
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("DistributedDataParallel")
        model = DDP(model, device_ids=[
            args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif n_gpu > 1:
        # model = torch.nn.DataParallel(model)
        model = DataParallelImbalance(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            # from apex.optimizers import FP16_Optimizer
            from pytorch_pretrained_bert.optimization_fp16 import FP16_Optimizer_State
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              # FusedAdam has been updated.  Simply initialize it identically to torch.optim.Adam, and call step() with no arguments.
                              bias_correction=False)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer_State(
                optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer_State(
                optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    if recover_step:
        logger.info("***** Recover optimizer: %d *****", recover_step)
        optim_recover = torch.load(os.path.join(
            args.output_dir, "optim.{0}.bin".format(recover_step)), map_location='cpu')
        if hasattr(optim_recover, 'state_dict'):
            optim_recover = optim_recover.state_dict()
        optimizer.load_state_dict(optim_recover)
        if args.loss_scale == 0:
            logger.info("***** Recover optimizer: dynamic_loss_scale *****")
            optimizer.dynamic_loss_scale = True

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", t_total)

        model.train
        if recover_step:
            start_epoch = recover_step + 1
        else:
            start_epoch = 1
        for i_epoch in trange(start_epoch, int(args.num_train_epochs) + 1, desc="Epoch",
                              disable=args.local_rank not in (-1, 0)):
            # 分佈式訓練
            if args.local_rank != -1:
                train_sampler.set_epoch(i_epoch)
            iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)',
                            disable=args.local_rank not in (-1, 0))
            for step, batch in enumerate(iter_bar):
                batch = [
                    t.to(device) if t is not None else None for t in batch]
                if args.has_sentence_oracle:
                    input_ids, segment_ids, input_mask, mask_qkv, lm_label_ids, masked_pos, masked_weights, is_next, task_idx, oracle_pos, oracle_weights, oracle_labels = batch
                else:
                    input_ids, segment_ids, input_mask, mask_qkv, lm_label_ids, masked_pos, masked_weights, is_next, task_idx, copy_labels, concept_relation = batch
                    # print('concept_relation',concept_relation)
                    oracle_pos, oracle_weights, oracle_labels = None, None, None

                # masked_weights 类似于input_mask，为了区分有效预测词和padding，分别标记为 1 和 0
                loss_tuple = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next, masked_pos=masked_pos,
                                   masked_weights=masked_weights, task_idx=task_idx, masked_pos_2=oracle_pos,
                                   masked_weights_2=oracle_weights,
                                   masked_labels_2=oracle_labels, mask_qkv=mask_qkv, knowledge_injection=kj.to(device),
                                   concept_relation=[rel_list[i] for i in concept_relation], copy_labels= copy_labels)

                # loss_tuple = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next, masked_pos=masked_pos, masked_weights=masked_weights, task_idx=task_idx, masked_pos_2=oracle_pos, masked_weights_2=oracle_weights,
                #                    masked_labels_2=oracle_labels, mask_qkv=mask_qkv,knowledge_injection=kj.to(device), concept_relation=[' '.join(i)for i in partition(rel_list,input_ids.shape[0])])
                masked_lm_loss, next_sentence_loss, copy_loss = loss_tuple
                if n_gpu > 1:  # mean() to average on multi-gpu.
                    # loss = loss.mean()
                    masked_lm_loss = masked_lm_loss.mean()
                    next_sentence_loss = next_sentence_loss.mean()
                # loss = masked_lm_loss + next_sentence_loss
                loss = masked_lm_loss + next_sentence_loss + 2 * copy_loss

                # logging for each step (i.e., before normalization by args.gradient_accumulation_steps)
                iter_bar.set_description('Iter (loss=%5.3f)' % loss.item())

                # ensure that accumlated gradients are normalized
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                    if amp_handle:
                        amp_handle._clear_cache()
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    lr_this_step = args.learning_rate * \
                                   warmup_linear(global_step / t_total,
                                                 args.warmup_proportion)
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                            pass
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Save a trained model
            if i_epoch == 30:
                if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                    logger.info(
                        "** ** * Saving fine-tuned model and optimizer ** ** * ")
                    model_to_save = model.module if hasattr(
                        model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(
                        args.output_dir, "model.{0}.bin".format(i_epoch))
                    torch.save(model_to_save.state_dict(), output_model_file)
                    output_optim_file = os.path.join(
                        args.output_dir, "optim.{0}.bin".format(i_epoch))
                    torch.save(optimizer.state_dict(), output_optim_file)

                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
