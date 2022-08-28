import os
import numpy as np
import math
import jieba
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import json


def get_k_fold_data(k, i, src, src_no_me, tgt, rel, req_stru):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    fold_size = math.ceil(len(src) / k)
    # fold_size = len(src) // k
    src_train, tgt_train, rel_train, src_no_me_train, req_stru_train = [], [], [], [], []
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        src_part, src_no_me_part, tgt_part, rel_part, req_stru_part = src[idx], src_no_me[idx], tgt[idx], rel[idx], req_stru[idx]
        if j == i:
            src_valid, src_no_me_vaild, tgt_valid, rel_vaild, req_stru_vaild = src_part, src_no_me_part, tgt_part, rel_part, req_stru_part
        elif src_train is [] or tgt_train is [] or rel_train is [] or src_no_me_train is [] or req_stru_train is []:
            src_train, src_no_me_train, tgt_train, rel_train, req_stru_train = src_part, src_no_me_part, tgt_part, rel_part, req_stru_part
        else:
            src_train.append(src_part)
            src_no_me_train.append(src_no_me_part)
            tgt_train.append(tgt_part)
            rel_train.append(rel_part)
            req_stru_train.append(req_stru_part)
    return src_train, src_no_me_train, tgt_train, rel_train, req_stru_train, src_valid, src_no_me_vaild, tgt_valid, rel_vaild, req_stru_vaild


def get_bleu_score(target, inference):
    # 分词
    target_fenci = ' '.join(jieba.cut(target))
    inference_fenci = ' '.join(jieba.cut(inference))

    # reference是标准答案 是一个列表，可以有多个参考答案，每个参考答案都是分词后使用split()函数拆分的子列表
    # # 举个reference例子
    # reference = [['this', 'is', 'a', 'duck']]
    reference = []  # 给定标准译文
    candidate = []  # 神经网络生成的句子
    # 计算BLEU
    reference.append(target_fenci.split())
    candidate = (inference_fenci.split())
    score1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    score2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
    score3 = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
    score4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))
    reference.clear()
    return score1, score2, score3, score4


if __name__ == "__main__":
    k_folk = 10
    rouge = Rouge()

    data_dir_path = 'uav_dataset/uav-annonation/'
    kfolk_dir_path = 'k_folk/'

    src_path = 'uav.src.txt'
    tgt_path = 'uav.tgt.txt'
    rel_path = 'uav.src.me.5hop.txt'
    src_no_me_path = 'uav.src.no.me.txt'
    req_structure_path = 'uav.structure.json'

    src_file = os.path.join(data_dir_path, src_path)
    src_no_file = os.path.join(data_dir_path, src_no_me_path)
    tgt_file = os.path.join(data_dir_path, tgt_path)
    rel_file = os.path.join(data_dir_path, rel_path)
    req_structure_file = os.path.join(data_dir_path, req_structure_path)

    inference_file_path = 'decoded_sentences/test/model.30.bin.test'

    kfolk_train_src_path = os.path.join(data_dir_path, kfolk_dir_path, src_path[:-4] + ".train.txt")
    kfolk_train_src_no_me_path = os.path.join(data_dir_path, kfolk_dir_path, src_no_me_path[:-4] + ".train.txt")
    kfolk_train_tgt_path = os.path.join(data_dir_path, kfolk_dir_path, tgt_path[:-4] + ".train.txt")
    kfolk_train_rel_path = os.path.join(data_dir_path, kfolk_dir_path, rel_path[:-4] + ".train.txt")
    kfolk_test_src_path = os.path.join(data_dir_path, kfolk_dir_path, src_path[:-4] + ".test.txt")
    kfolk_test_src_no_me_path = os.path.join(data_dir_path, kfolk_dir_path, src_no_me_path[:-4] + ".test.txt")
    kfolk_test_tgt_path = os.path.join(data_dir_path, kfolk_dir_path, tgt_path[:-4] + ".test.txt")
    kfolk_test_rel_path = os.path.join(data_dir_path, kfolk_dir_path, rel_path[:-4] + ".test.txt")
    kfolk_train_req_structure_path = os.path.join(data_dir_path, kfolk_dir_path, req_structure_path[:-5] + ".train.json")
    kfolk_test_req_structure_path = os.path.join(data_dir_path, kfolk_dir_path, req_structure_path[:-5] + ".test.json")
    print(kfolk_train_src_path)
    print(kfolk_train_tgt_path)
    print(kfolk_train_rel_path)
    print(kfolk_train_src_no_me_path)

    with open(src_file, "r", encoding='utf-8') as f_src, open(src_no_file, "r", encoding='utf-8') as f_src_no_me, open(
            tgt_file, "r", encoding='utf-8') as f_tgt, open(rel_file, "r", encoding='utf-8') as f_rel,\
            open(req_structure_file, 'r' ,encoding='utf-8') as f_stru:
        src_list = []
        src_no_me_list = []
        tgt_list = []
        rel_list = []
        for src in f_src.readlines():
            src_list.append(src)
        for src_no_me in f_src_no_me.readlines():
            src_no_me_list.append(src_no_me)
        for tgt in f_tgt.readlines():
            tgt_list.append(tgt)
        for rel in f_rel.readlines():
            rel_list.append(rel)
        ref_json = json.load(f_stru)


    bleu1_score = 0
    bleu2_score = 0
    bleu3_score = 0
    bleu4_score = 0
    bleu_num = 0

    rouge1_r_score, rouge1_p_score, rouge1_f_score = 0, 0, 0
    rouge2_r_score, rouge2_p_score, rouge2_f_score = 0, 0, 0
    rougel_r_score, rougel_p_score, rougel_f_score = 0, 0, 0

    for i in range(k_folk):
        print('================================', i, '==================================')
        f_kfolk_train_src = open(kfolk_train_src_path, 'w')
        f_kfolk_train_src_no_me = open(kfolk_train_src_no_me_path, 'w')
        f_kfolk_train_tgt = open(kfolk_train_tgt_path, 'w')
        f_kfolk_train_rel = open(kfolk_train_rel_path, 'w')
        f_kfolk_train_stru = open(kfolk_train_req_structure_path,'w')
        f_kfolk_test_src = open(kfolk_test_src_path, 'w')
        f_kfolk_test_src_no_me = open(kfolk_test_src_no_me_path, 'w')
        f_kfolk_test_tgt = open(kfolk_test_tgt_path, 'w')
        f_kfolk_test_rel = open(kfolk_test_rel_path, 'w')
        f_kfolk_test_stru = open(kfolk_test_req_structure_path, 'w')
        src_folk_list, src_no_me_folk_list, tgt_folk_list, rel_folk_list, req_stru_folk_list, src_folk_valid, src_no_me_folk_valid, tgt_folk_valid, rel_folk_vaild, req_stru_folk_valid = get_k_fold_data(
            k_folk, i, src_list, src_no_me_list, tgt_list, rel_list, ref_json)
        src_folk_list = sum(src_folk_list, [])
        src_no_me_folk_list = sum(src_no_me_folk_list, [])
        tgt_folk_list = sum(tgt_folk_list, [])
        rel_folk_list = sum(rel_folk_list, [])
        req_stru_folk_list = sum(req_stru_folk_list, [])
        # print('src_folk_list',src_folk_list)
        for j in src_folk_list:
            f_kfolk_train_src.write(j)
        for j in src_no_me_folk_list:
            f_kfolk_train_src_no_me.write(j)
        for j in tgt_folk_list:
            f_kfolk_train_tgt.write(j)
        for j in rel_folk_list:
            f_kfolk_train_rel.write(j)
        # for j in req_stru_folk_list:
        json.dump(req_stru_folk_list, f_kfolk_train_stru)
            # f_kfolk_train_stru.write(str(j))
        for j in src_folk_valid:
            f_kfolk_test_src.write(j)
        for j in src_no_me_folk_valid:
            f_kfolk_test_src_no_me.write(j)
        for j in tgt_folk_valid:
            f_kfolk_test_tgt.write(j)
        for j in rel_folk_vaild:
            f_kfolk_test_rel.write(j)
        json.dump(req_stru_folk_valid, f_kfolk_test_stru)
        f_kfolk_train_src.close()
        f_kfolk_train_src_no_me.close()
        f_kfolk_train_tgt.close()
        f_kfolk_train_rel.close()
        f_kfolk_train_stru.close()
        f_kfolk_test_src.close()
        f_kfolk_test_src_no_me.close()
        f_kfolk_test_tgt.close()
        f_kfolk_test_rel.close()
        f_kfolk_test_stru.close()

        '''1.    model.50.bin.20220501.time.uav.sqrted.nodense.mask.1by1.test'''
        status = os.system(
            'cd /root/Desktop/uav_req_gen_unilm_v2/tmp/new_finetuned_models && rm -rf bert_save && mkdir bert_save && cd /root/Desktop/uav_req_gen_unilm_v2')
        # status = os.system('rm -rf bert_save ')
        # status = os.system('mkdir bert_save')
        # status = os.system('cd /root/Desktop/uav_req_gen_unilm_v2')

        status = os.system('bash train.sh')

        status = os.system('CUDA_VISIBLE_DEVICES=0,1,2,3 bash test.sh 30 test')
        status = os.system('cd /root/Desktop/uav_req_gen_unilm_v2/tmp/new_finetuned_models')
        status = os.system(
            'cd /root/Desktop/uav_req_gen_unilm_v2/tmp/new_finetuned_models && rm -rf bert_save && mkdir bert_save && cd /root/Desktop/uav_req_gen_unilm_v2')
        # status = os.system('rm -r /bert_save ')
        # status = os.system('mkdir bert_save')
        # status = os.system('cd /root/Desktop/uav_req_gen_unilm_v2')

        inference_file = open(inference_file_path, 'r', encoding='utf-8').readlines()

        for j in inference_file:
            bleu_num += 1
            if len(j) > 5:
                bleu1, bleu2, bleu3, bleu4 = get_bleu_score(tgt_folk_valid[inference_file.index(j)], j)
                bleu1_score += bleu1
                bleu2_score += bleu2
                bleu3_score += bleu3
                bleu4_score += bleu4

                print('==============================================')
                print('第',inference_file.index(j),'句話')
                print(tgt_folk_valid[inference_file.index(j)])
                print(j)

                print('bleu1_score', bleu1)
                print('bleu2_score', bleu2)
                print('bleu3_score', bleu3)
                print('bleu4_score', bleu4)

                rouge_score = rouge.get_scores(tgt_folk_valid[inference_file.index(j)], j)

                print(rouge_score)

                r1_r, r1_p, r1_f = rouge_score[0]["rouge-1"]['r'], rouge_score[0]["rouge-1"]['p'], \
                                   rouge_score[0]["rouge-1"]['f']
                r2_r, r2_p, r2_f = rouge_score[0]["rouge-2"]['r'], rouge_score[0]["rouge-2"]['p'], \
                                   rouge_score[0]["rouge-2"]['f']
                rl_r, rl_p, rl_f = rouge_score[0]["rouge-l"]['r'], rouge_score[0]["rouge-l"]['p'], \
                                   rouge_score[0]["rouge-l"]['f']
                rouge1_r_score += r1_r
                rouge1_p_score += r1_p
                rouge1_f_score += r1_f
                rouge2_r_score += r2_r
                rouge2_p_score += r2_p
                rouge2_f_score += r2_f
                rougel_r_score += rl_r
                rougel_p_score += rl_p
                rougel_f_score += rl_f

                print("rouge-1", rouge_score[0]["rouge-1"])
                print("rouge-2", rouge_score[0]["rouge-2"])
                print("rouge-l", rouge_score[0]["rouge-l"])

                print('==============================================')

        total_bleu1_score = bleu1_score / bleu_num
        print('total_bleu1_score========', total_bleu1_score)
    bleu1_score = bleu1_score / 99
    bleu2_score = bleu2_score / 99
    bleu3_score = bleu2_score / 99
    bleu4_score = bleu2_score / 99

    rouge1_r_score = rouge1_r_score / 99
    rouge1_p_score = rouge1_p_score / 99
    rouge1_f_score = rouge1_f_score / 99
    rouge2_r_score = rouge2_r_score / 99
    rouge2_p_score = rouge2_p_score / 99
    rouge2_f_score = rouge2_f_score / 99
    rougel_r_score = rougel_r_score / 99
    rougel_p_score = rougel_p_score / 99
    rougel_f_score = rougel_f_score / 99

    print('total_bleu1_score', bleu1_score)
    print('total_bleu2_score', bleu2_score)
    print('total_bleu3_score', bleu3_score)
    print('total_bleu4_score', bleu4_score)

    print('total_rouge1_r_score', rouge1_r_score)
    print('total_rouge1_p_score', rouge1_p_score)
    print('total_rouge1_f_score', rouge1_f_score)
    print('total_rouge2_r_score', rouge2_r_score)
    print('total_rouge2_p_score', rouge2_p_score)
    print('total_rouge2_f_score', rouge2_f_score)
    print('total_rougel_r_score', rougel_r_score)
    print('total_rougel_p_score', rougel_p_score)
    print('total_rougel_f_score', rougel_f_score)
