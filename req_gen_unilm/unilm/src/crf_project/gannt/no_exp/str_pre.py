import stanza

class structure_pre:
    def __init__(self, req_list):
        self.req_list = req_list
        self.nlp = stanza.Pipeline('en', processors='tokenize, pos, lemma, depparse')
        self.open_list = ['gannt']
        self.file_path = '/root/Desktop/uav_req_gen_unilm_v2/unilm/src/crf_project'
        # self.init_nlp()

    def operation_pre(self):
        conditions = ["if", "when"]

        for open_tag in self.open_list:
            res = []
            # 此处打开需求集
            # file = open(file_path + '/json_data/' + open_tag + '_data.txt')
            # fileJson = file.readlines()
            whether_main = 1
            for i, one in enumerate(self.req_list):
                line = one[0:-1]
                doc = self.nlp(line)
                for j, sent in enumerate(doc.sentences):
                    for word in sent.words:
                        if word.lemma in conditions:
                            whether_main = 0  # 0代表主句， 1代表从句
                        if whether_main == 1 and 'V' in word.xpos:
                            tag = 1
                        else:
                            tag = 0
                        if whether_main == 0 and (word.lemma == "," or word.lemma == "."):
                            whether_main = 1
                        res.append(f'{word.text}\t{word.xpos}\t{word.deprel}\t{tag}\n')
                res.append(" \n")

            # 此处写入operation的预处理文件
            fw = open(self.file_path + '/exp/pre/operation_pre_' + open_tag + '.txt', 'w')
            for r in res:
                fw.write(r)
            fw.close()

        print("operation pre done")


    def agent_pre(self):
        for open_tag in self.open_list:
            res = []

            # 此处打开需求集
            # file = open(file_path + '/json_data/' + open_tag + '_data.txt')
            # fileJson = file.readlines()

            flag = 0
            whether_main = 1
            operation = ""
            for i, one in enumerate(self.req_list):
                line = one[0:-1]
                doc = self.nlp(line)
                for j, sent in enumerate(doc.sentences):
                    for word in sent.words:
                        tag = "O"
                        res.append(f'{word.text}\t{word.xpos}\t{word.deprel}\t{tag}')
                res.append(" \n")

            # 此处打开operation的预处理文件
            operation_file = open(self.file_path + '/exp/pre/operation_pre_' + open_tag + '.txt')
            rl = operation_file.readlines()
            for i, line in enumerate(res):
                lline = line.split()
                if len(lline) > 3:
                    tagg = lline[-1]
                    rl[i] = rl[i][0:-1]
                    rl[i] += f"\t{tagg}\n"

            # 此处写入agent的预处理文件
            fw = open(self.file_path + '/exp/pre/agent_pre_' + open_tag + '.txt', 'w')
            for r in rl:
                fw.write(r)
            fw.close()

        print("agent pre done")


    def input_pre(self):
        for open_tag in self.open_list:
            res = []
            requirements = []
            # 此处打开需求集
            # file = open(file_path + '/json_data/' + open_tag + '_data.txt')
            # fileJson = file.readlines()
            for i, one in enumerate(self.req_list):
                line = one[0:-1]
                requirements.append(line)

            # 此处打开operation预处理文件
            operation_file = open(self.file_path + '/exp/pre/operation_pre_' + open_tag + '.txt')
            rl = operation_file.readlines()
            pos = 0
            for i, one in enumerate(requirements):
                doc = self.nlp(one)
                for sent in doc.sentences:
                    words = sent.words
                    one_sent = [i.text.lower() for i in words]  # 根据stanza分出来的words拼成sentence的列表
                    tag_list = ["O" for i in range(len(one_sent))]  # 标记列表
                    for q, word in enumerate(sent.words):
                        opLine = rl[pos].split()
                        opTag = opLine[-1]
                        pos += 1
                        res.append(f'{word.text}\t{word.xpos}\t{word.deprel}\t{opTag}\t{tag_list[q]}\n')
                res.append(" \n")
                pos += 1

            # 此处写入input预处理文件
            fw = open(self.file_path + '/exp/pre/input_pre_' + open_tag + '.txt', 'w')
            for r in res:
                fw.write(r)
            fw.close()

        print("input pre done")


    def output_pre(self):
        output_num = 0
        for open_tag in self.open_list:
            res = []
            # 此处打开需求集
            # file = open(file_path + '/json_data/' + open_tag + '_data.txt')
            # fileJson = file.readlines()
            requirements = []
            for i, one in enumerate(self.req_list):
                line = one[0:-1]
                requirements.append(line)

            # 此处打开operation预处理文件
            operation_file = open(self.file_path +'/exp/pre/operation_pre_' + open_tag + '.txt')
            rl = operation_file.readlines()
            pos = 0
            # 读入operation_pre.txt 加入operation字段
            for i, one in enumerate(requirements):
                doc = self.nlp(one)
                for sent in doc.sentences:
                    words = sent.words
                    one_sent = [i.text.lower() for i in words]  # 根据stanza分出来的words拼成sentence的列表
                    tag_list = ["O" for i in range(len(one_sent))]  # 标记列表

                    for q, word in enumerate(sent.words):
                        opLine = rl[pos].split()
                        opTag = opLine[-1]
                        pos += 1
                        res.append(f'{word.text}\t{word.xpos}\t{word.deprel}\t{opTag}\t{tag_list[q]}\n')
                res.append(" \n")
                pos += 1

            # 此处写入output预处理文件
            fw = open(self.file_path + '/exp/pre/output_pre_' + open_tag + '.txt', 'w')
            for r in res:
                fw.write(r)
            fw.close()
        print("output pre done")


    def restriction_pre(self):
        restriction_num = 0

        for open_tag in self.open_list:
            res = []

            # 此处打开需求集
            # file = open(file_path + '/json_data/' + open_tag + '_data.txt')
            # fileJson = file.readlines()
            requirements = []
            for i, one in enumerate(self.req_list):
                line = one[0:-1]
                requirements.append(line)

            for i, one in enumerate(requirements):
                doc = self.nlp(one)
                for sent in doc.sentences:
                    words = sent.words
                    one_sent = [i.text.lower() for i in words]  # 根据stanza分出来的words拼成sentence的列表
                    tag_list = ["O" for i in range(len(one_sent))]  # 标记列表
                    for q, word in enumerate(sent.words):
                        if word.text.lower() == "then" or word.text.lower() == "only":
                            opTag = 1
                        else:
                            opTag = 0
                        res.append(f'{word.text}\t{word.xpos}\t{word.deprel}\t{opTag}\t{tag_list[q]}\n')
                res.append(" \n")

            # 此处写入restriction预处理文件
            fw = open(self.file_path + '/exp/pre/restriction_pre_' + open_tag + '.txt', 'w')
            for r in res:
                fw.write(r)
            fw.close()

        print("restriction pre done")


    def event_pre(self):
        for open_tag in self.open_list:
            res = []

            # 此处打开需求集
            # file = open(file_path + '/json_data/' + open_tag + '_data.txt')
            # fileJson = file.readlines()
            flag = 0
            requirements = []
            events = []
            for i, one in enumerate(self.req_list):
                line = one[0:]
                con_sent = []
                con_sent_when = ""
                con_sent_if = ""
                if "when " in line.lower() or "if " in line.lower():
                    if_idx = -1
                    when_idx = -1
                    if "when " in line.lower():
                        when_idx = line.lower().find("when ")
                        t = when_idx
                        while line[t] != ',' and line[t] != '.':
                            con_sent_when += line[t]
                            t += 1
                    if "if " in line.lower():
                        if_idx = line.lower().find("if ")
                        t = if_idx
                        while line[t] != ',' and line[t] != '.':
                            con_sent_if += line[t]
                            t += 1
                    if when_idx > if_idx:
                        if if_idx != -1:
                            con_sent.append(con_sent_if)
                            if con_sent_when not in con_sent_if:
                                con_sent.append(con_sent_when)
                        else:
                            con_sent.append(con_sent_when)
                    elif if_idx > when_idx:
                        if when_idx != -1:
                            con_sent.append(con_sent_when)
                            if con_sent_if not in con_sent_when:
                                con_sent.append(con_sent_if)
                        else:
                            con_sent.append(con_sent_if)

                events.append(con_sent)

                requirements.append(line)

            res_agent = []
            res_operation = []
            res_input = []
            res_output = []
            res_restriction = []
            cnt = 0
            for i, one_list in enumerate(events):
                if one_list != []:
                    cnt += 1
                    one = one_list[0]
                    for j in one_list[1:]:
                        one += " " + j
                    doc = self.nlp(one)
                    sent = doc.sentences[0]
                    words = sent.words
                    one_sent = [i.text.lower() for i in words]  # 根据\stanza分出来的words拼成sentence的列表
                    op_sent = [i.lemma.lower() for i in words]  # operation需要根据原形识别
                    tag_list_agent = ["O" for i in range(len(one_sent))]
                    tag_list_operation = [0 for i in range(len(one_sent))]
                    tag_list_input = ["O" for i in range(len(one_sent))]
                    tag_list_output = ["O" for i in range(len(one_sent))]
                    tag_list_restriction = ["O" for i in range(len(one_sent))]  # 标记列表
                    event = {
                        "agent": {"entity": ""},
                        "operation": {"operation": ""},
                        "input": {"()": []},
                        "output": {"()": []},
                        "restriction": {"()": []}
                    }

                    for q, word in enumerate(sent.words):
                        if word.text.lower() == "then" or word.text.lower() == "only":
                            restrictionTag = 1
                        else:
                            restrictionTag = 0
                        if 'V' in word.xpos:
                            tag_list_operation[q] = 1
                        res_input.append(
                            f'{word.text}\t{word.xpos}\t{word.deprel}\t{tag_list_operation[q]}\t{tag_list_input[q]}\n')
                        res_output.append(
                            f'{word.text}\t{word.xpos}\t{word.deprel}\t{tag_list_operation[q]}\t{tag_list_output[q]}\n')
                        res_operation.append(f'{word.text}\t{word.xpos}\t{word.deprel}\t{tag_list_operation[q]}\n')
                        res_restriction.append(
                            f'{word.text}\t{word.xpos}\t{word.deprel}\t{restrictionTag}\t{tag_list_restriction[q]}\n')
                        res_agent.append(
                            f'{word.text}\t{word.xpos}\t{word.deprel}\t{tag_list_operation[q]}\t{tag_list_agent[q]}\n')
                    res_agent.append(" \n")
                    res_restriction.append(" \n")
                    res_output.append(" \n")
                    res_input.append(" \n")
                    res_operation.append(" \n")

            # 此处写入event五元组的预处理文件
            fw = open(self.file_path + '/exp/pre/event_pre_agent_' + open_tag + '.txt', 'w')
            ft = open(self.file_path + '/exp/pre/event_pre_operation_' + open_tag + '.txt', 'w')
            fp = open(self.file_path + '/exp/pre/event_pre_input_' + open_tag + '.txt', 'w')
            fq = open(self.file_path + '/exp/pre/event_pre_output_' + open_tag + '.txt', 'w')
            fe = open(self.file_path + '/exp/pre/event_pre_restriction_' + open_tag + '.txt', 'w')
            for r in res_agent:
                fw.write(r)
            for r in res_operation:
                ft.write(r)
            for r in res_input:
                fp.write(r)
            for r in res_output:
                fq.write(r)
            for r in res_restriction:
                fe.write(r)
            fw.close()
            ft.close()
            fp.close()
            fq.close()
            fe.close()

        print("event pre done")

    def run_str_pre(self):
        self.operation_pre()
        self.agent_pre()
        self.input_pre()
        self.output_pre()
        self.restriction_pre()
        self.event_pre()