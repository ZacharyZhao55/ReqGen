import stanza

nlp = stanza.Pipeline('en', processors='tokenize, pos, lemma, depparse')
file_path = '/root/Desktop/crf_project/crf_project'
# open_list = ['uav', 'pure', 'gannt', 'worldvista']
open_list = ['gannt']



for open_tag in open_list:
	res = []

	# 此处打开需求集
	file = open(file_path+'/json_data/'+open_tag+'_data.txt')
	fileJson = file.readlines()
	flag = 0
	requirements = []
	events = []
	for i, one in enumerate(fileJson):
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
			doc = nlp(one)
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
	fw = open(file_path+'/exp/pre/event_pre_agent_'+open_tag+'.txt', 'w')
	ft = open(file_path+'/exp/pre/event_pre_operation_'+open_tag+'.txt', 'w')
	fp = open(file_path+'/exp/pre/event_pre_input_'+open_tag+'.txt', 'w')
	fq = open(file_path+'/exp/pre/event_pre_output_'+open_tag+'.txt', 'w')
	fe = open(file_path+'/exp/pre/event_pre_restriction_'+open_tag+'.txt', 'w')
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
