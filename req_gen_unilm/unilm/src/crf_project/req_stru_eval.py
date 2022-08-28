import json
import os
from crf_project.gannt.no_exp.str_pre import structure_pre
from RS4RE import RS4RE
import numpy as np

class MySentencesNotCut(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        fname_list = []
        for fname in os.listdir(self.dirname):
            fname_list.append(fname)
        fname_list.sort(key=lambda x: int(x[x.find("-")+1:-4]))
        for fname in fname_list:
            print(fname)
            for line in open(os.path.join(self.dirname,fname),encoding='utf-8'):
                yield line


def req_structurize(req_list):
    cands_req_list = []
    for r in req_list:
        if '.' not in r[-3:]:
            r += '.'
        if len(r) < 3:
            cands_req_list.append('###################')
        else:
            cands_req_list.append(r)
    print('cands_req_list',len(cands_req_list))
    structure_pre(cands_req_list).run_str_pre()
    os.system('bash /root/Desktop/uav_req_gen_unilm_v2/unilm/src/crf_project/gannt/auto_gannt.sh')
    os.system('python /root/Desktop/uav_req_gen_unilm_v2/unilm/src/crf_project/gannt/generate_tuple.py')

def req_stru_eval():
    gen_req_stru_path = 'crf_project/gannt/external/test_gannt.json'
    req_stru_path = '../../uav_dataset/uav-annonation/uav.structure.json'
    with open(gen_req_stru_path, 'r', encoding='utf-8') as fw_gen:
        ref_json = json.load(fw_gen)
    gen_re2st_list = RS4RE.json2dict_gen(ref_json)

    with open(req_stru_path, 'r', encoding='utf-8') as fw_ref:
        ref_json = json.load(fw_ref)
    ref_re2st_list = RS4RE.json2dict_gen(ref_json)
    print(ref_re2st_list)

    stru_score = []
    print('gen_re2st_list',len(gen_re2st_list))
    print('ref_re2st_list', len(ref_re2st_list))
    for i in range(len(gen_re2st_list)):
        score_1gram, score_2gram = RS4RE.main(gen_re2st_list[i], ref_re2st_list[i])
        # if max_score < score_1gram+score_2gram:
        score = score_1gram+score_2gram
        stru_score.append(score)
            # best_b = i
    return stru_score

inference_dir_path = '/root/Desktop/uav_req_gen_unilm_v2/inference/'
inference_path_list = ['gpt3']
reference_path = '/root/Desktop/uav_req_gen_unilm_v2/uav_dataset/uav-annonation/uav.tgt.txt'

ref_list = []
with open(reference_path,encoding='utf-8') as ref_file:
    for i in ref_file.readlines():
        # print(i)
        ref_list.append(i)
for i in inference_path_list:
    print('=====================',i)
    inference_path = os.path.join(inference_dir_path,i)
    mysentence = MySentencesNotCut(inference_path)
    inf_list = []
    for j in mysentence:
        inf_list.append(j)
    req_structurize(inf_list)
    stru_score = req_stru_eval()
    print(stru_score)
    s = 0
    for n in stru_score:
        s += n
    print(s/93)