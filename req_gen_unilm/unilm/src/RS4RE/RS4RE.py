import math
import collections
import json
from rouge import Rouge


class RS4RE():
    def __init__(self, req_inf, req_ref):
        self.req_inf = req_inf
        self.req_ref = req_ref

    def get_alpha_value(self,re2st_word_list,req_structure):
        words_frequency = collections.Counter(re2st_word_list).most_common()
        alpha_dict = {}
        for key, value in req_structure.items():
            if len(value) != 0:
                count_fre = 0
                if type(value) == str:
                    value_list = value.split(' ')
                elif type(value[0]) == dict:
                    value_list = []
                    for v in value:
                        value_list.extend(v['entity'].split(' '))
                else:
                    value_list = []
                    for v in value:
                        value_list.extend(v.split(' '))
                for word in value_list:
                    for word_f in words_frequency:
                        if word in word_f:
                            count_fre += math.log(word_f[1] + 1, 2)
                            # count_fre += word_f[1]
                alpha = len(value_list) / count_fre
            else:
                alpha = 1
            alpha_dict[key] = alpha
        return alpha_dict

    def get_n_gram_value(self,inf_ele,ref_ele):
        rouge = Rouge()
        if inf_ele == '':
            return 0,0,0
        rouge_score = rouge.get_scores(inf_ele, ref_ele)

        r1_r, r1_p, r1_f = rouge_score[0]["rouge-1"]['r'], rouge_score[0]["rouge-1"]['p'], \
                           rouge_score[0]["rouge-1"]['f']
        r2_r, r2_p, r2_f = rouge_score[0]["rouge-2"]['r'], rouge_score[0]["rouge-2"]['p'], \
                           rouge_score[0]["rouge-2"]['f']
        rl_r, rl_p, rl_f = rouge_score[0]["rouge-l"]['r'], rouge_score[0]["rouge-l"]['p'], \
                           rouge_score[0]["rouge-l"]['f']

        return r1_r, r2_r, rl_r

    def get_word_list(self,re2st):
        re2st_list = []
        re2st_v = re2st.values()
        re2st_str = ''
        for st in re2st_v:
            if type(st) == list and len(st) != 0:
                for ele in st:
                    if type(ele) == str:
                        re2st_str += ele
                        re2st_list.append(ele)
                    elif type(ele) == list:
                        re2st_str += ele['entity']
                        re2st_list.append(ele['entity'])
                    elif type(ele) == dict:
                        re2st_str += ' '.join(ele.values())
                        re2st_list.append(re2st_str)
                    re2st_str += ' '
            elif type(st) == str:
                re2st_str += st
                re2st_list.append(st)
            re2st_str += ' '
        re2st_word_list = re2st_str.split(' ')
        return re2st_word_list,re2st_list

    def get_req_st_dict(self,req_st):
        ele = {}
        for key, value in req_st.items():
            if len(value) != 0:
                if type(value) == str:
                    ele[key] = value
                elif type(value[0]) == dict:
                    ele_str = ''
                    for v in value:
                        ele_str += v['entity']
                        ele[key] = ele_str
                else:
                    ele_str = ''
                    for v in value:
                        ele_str += v
                        ele[key] = ele_str
            else:
                ele[key] = ''
        return ele

    def run_rs4re(self):
        inf_word_list, inf_list = self.get_word_list(self.req_inf)
        ref_word_list, ref_list = self.get_word_list(self.req_ref)

        inf_alpha_dict = self.get_alpha_value(inf_word_list,self.req_inf)
        ref_alpha_dict = self.get_alpha_value(ref_word_list,self.req_ref)


        inf_ele_dict = self.get_req_st_dict(self.req_inf)
        ref_ele_dict = self.get_req_st_dict(self.req_ref)

        score_1gram = 0
        score_2gram = 0
        num = 0
        for inf_key, inf_value in inf_ele_dict.items():
            for ref_key, ref_value in ref_ele_dict.items():
                if inf_key == ref_key:
                    if ref_value != '':
                        # print('+++++++++++++',ref_value,'0000000000',inf_value)
                        if inf_key == 'agent' or inf_key == 'event' or inf_key == 'opertaion':
                                num += 1
                                r1_r, r2_r, rl_r = self.get_n_gram_value(inf_value,ref_value)
                                score_1gram += inf_alpha_dict[ref_key] / ref_alpha_dict[ref_key] * r1_r
                                score_1gram *= 0.25
                                score_2gram += inf_alpha_dict[ref_key] / ref_alpha_dict[ref_key] * r2_r
                                score_2gram *= 0.25
                        else:
                                num += 1
                                r1_r, r2_r, rl_r = self.get_n_gram_value(inf_value,ref_value)
                                score_1gram += inf_alpha_dict[ref_key] / ref_alpha_dict[ref_key] * r1_r
                                score_1gram *= 0.12
                                score_2gram += inf_alpha_dict[ref_key] / ref_alpha_dict[ref_key] * r2_r
                                score_2gram *= 0.12

                    # score_ngram += r1_r
        if num == 0:
            return 0,0

        return score_1gram / num, score_2gram / num

def json2dict_gen(req_structure_json):
    re2st_list = []
    for i in req_structure_json:
        re2st = {}
        try:
            re2st['agent'] = i['agent']
        except:
            re2st['agent'] = ''
        try:
            re2st['operation'] = i['operation']
        except:
            re2st['operation'] = ''
        try:
            re2st['input'] = i['input']
        except:
            re2st['input'] = []
        try:
            re2st['output'] = i['output']
        except:
            re2st['output'] = []
        try:
            re2st['restriction'] = i['restriction']
        except:
            re2st['restriction'] = []
        try:
            if i.has_key['event']:
                event_str = ''
                if i['event'][0].has_key('agent'):
                    event_agent = i['event'][0]['agent']
                    event_str += event_agent
                if i['event'][0].has_key('operation'):
                    event_operation = i['event'][0]['operation']
                    event_str += event_operation
                if i['event'][0].has_key('restriction'):
                    event_restriction = i['event'][0]['restriction']
                    event_str += ' '.join(event_restriction)
                if i['event'][0].has_key('input'):
                    event_input = i['event'][0]['input']
                    event_str += ' '.join(event_input)
                if i['event'][0].has_key('output'):
                    event_output = i['event'][0]['output']
                    event_str += ' '.join(event_output)
                event_str['event'] = event_str
        except:
            re2st['event'] = []
        re2st_list.append(re2st)
    return re2st_list

def json2dict(req_structure_json):
    re2st_list = []
    for i in req_structure_json:
        re2st = {}
        try:
            # re2st['id'] = i['#']
            # re2st['req_name'] = i[':']
            re2st['agent'] = i['agent']['entity']
        except:
            re2st['agent'] = ''
        try:
            re2st['operation'] = i['operation']['operation']
        except:
            re2st['operation'] = ''
        try:
            re2st['input'] = i['input']['()']
        except:
            re2st['input'] = []
        try:
            re2st['output'] = i['output']['()']
        except:
            re2st['output'] = []
        try:
            re2st['restriction'] = i['restriction']['()']
        except:
            re2st['restriction'] = []
        try:
            if i.has_key['event']:
                event_str = ''
                if i['event'].has_key('agent'):
                    event_agent = i['event']['()'][0]['agent']['entity']
                    event_str += event_agent
                if i['event'].has_key('operation'):
                    event_operation = i['event']['()'][0]['operation']['operation']
                    event_str += event_operation
                if i['event'].has_key('restriction'):
                    event_restriction = i['event']['()'][0]['restriction']['()']
                    event_str += ' '.join(event_restriction)
                if i['event'].has_key('input'):
                    event_input = i['event']['()'][0]['input']['()']
                    event_str += ' '.join(event_input)
                if i['event'].has_key('output'):
                    event_output = i['event']['()'][0]['output']['()']
                    event_str += ' '.join(event_output)
                event_str['event'] = event_str
        except:
            re2st['event'] = []
        re2st_list.append(re2st)
    return re2st_list

def main(req_ref, req_inf):
    rs4re = RS4RE(req_inf,req_ref)
    score_1gram, score_2gram = rs4re.run_rs4re()
    return score_1gram, score_2gram

if __name__ == "__main__":
    ref_json_path = 'gannt_requirement_structure.json'
    with open(ref_json_path, 'r', encoding='utf-8') as fw:
        ref_json = json.load(fw)
    ref_re2st_list = json2dict(ref_json)

    print(ref_re2st_list[0])

    for i in ref_re2st_list:
        print(i)
        score_1gram,score_2gram = main(i,i)
        print(score_1gram)
        print(score_2gram)
