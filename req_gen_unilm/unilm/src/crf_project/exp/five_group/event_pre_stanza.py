import json, stanza, re

nlp = stanza.Pipeline('en', processors='tokenize, pos, lemma, depparse')
file_path = '/Users/tuanz_lu/PycharmProjects/Pytorch-learning'


def deal_agent(deal_sent, agentList):
	agent_list = ["O" for i in range(len(deal_sent))]
	agents = agentList.split('|')
	for agent in agents:
		one_inp = agent.lower().split()  # 每个input的分词列表
		ll = len(one_inp)
		for j in range(len(deal_sent)):
			if deal_sent[j:j+ll] == one_inp and one_inp != []:
				agent_list[j] = "B"
				if ll != 1:
					agent_list[j+ll-1] = "E"
				m = j+1
				while m < j+ll-1:
					agent_list[m] = "I"
					m += 1
				break
	return agent_list

def deal_operation(deal_sent, opList):
	op_list = [0 for i in range(len(deal_sent))]
	ops = opList.split('|')
	for op in ops:
		one_op = op.lower().split()
		ll = len(one_op)
		for j in range(len(deal_sent)):
			if deal_sent[j:j+ll] == one_op:
				m = j
				while m <= j + ll - 1:
					op_list[m] = 1
					m += 1
				break
	return op_list


def deal_input(deal_sent, inList):
	in_list = ["O" for i in range(len(deal_sent))]
	for inp in inList:
		inp["entity"] = inp["entity"].replace('/', ' / ')
		inp["entity"] = inp["entity"].replace('-', ' - ')
		one_inp = inp["entity"].lower().split()  # 每个input的分词列表
		ll = len(one_inp)

		for j in range(len(deal_sent)):
			if deal_sent[j:j+ll] == one_inp:
				in_list[j] = "B"
				if ll != 1:
					in_list[j+ll-1] = "E"
				m = j+1
				while m < j+ll-1:
					in_list[m] = "I"
					m += 1
				break
	return in_list


def deal_output(deal_sent, outList):
	out_list = ["O" for i in range(len(deal_sent))]
	for inp in outList:
		inp["entity"] = inp["entity"].replace('/', ' / ')
		one_inp = inp["entity"].lower().split()  # 每个output的分词列表
		ll = len(one_inp)
		for j in range(len(deal_sent)):
			if deal_sent[j:j+ll] == one_inp:
				out_list[j] = "B"
				if ll != 1:
					out_list[j+ll-1] = "E"
				m = j+1
				while m < j+ll-1:
					out_list[m] = "I"
					m += 1
				break
	return out_list


def deal_restriction(deal_sent, resList):
	res_list = ["O" for i in range(len(deal_sent))]
	for inp in resList:
		inp = inp.replace('-', " - ")
		one_inp = inp.lower().split()  # 每个output的分词列表
		ll = len(one_inp)
		for j in range(len(deal_sent)):
			if deal_sent[j:j+ll] == one_inp:
				res_list[j] = "B"
				if ll != 1:
					res_list[j+ll-1] = "E"
				m = j+1
				while m < j+ll-1:
					res_list[m] = "I"
					m += 1
				break
	return res_list


open_list = ['E', 'A', 'B', 'C', 'D']


for open_tag in open_list:
	res = []

	file = open(file_path+'/json_data/five/'+open_tag+'_data.json')
	fileJson = json.load(file)
	flag = 0
	requirements = []
	events = []
	eventList = []
	eventList_cur = []
	for i, one in enumerate(fileJson):
		if flag > i:
			inputList_cur = []
			continue
		else:
			line = one[":"]
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
			if "event" in one:
				eventList_cur = one["event"]["()"]
			flag = i + 1
			while flag < len(fileJson) and fileJson[flag][":"] == line:  # 合并同一个需求的operation
				flag += 1
			eventList.append(eventList_cur)
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
			if len(eventList[i]) == 1:
				event = eventList[i][0]
			else:
				for j in eventList[i]:
					if "agent" in j:
						if j["agent"]["entity"] not in event["agent"]["entity"]:
							event["agent"]["entity"] += j["agent"]["entity"] + '|'  # 用|分割
					if "operation" in j:
						event["operation"]["operation"] += j["operation"]["operation"] + '|'
					if "input" in j:
						for in_one in j["input"]["()"]:
							event["input"]["()"].append(in_one)
					if "output" in j:
						for in_one in j["output"]["()"]:
							event["output"]["()"].append(in_one)
					if "restriction" in j:
						for in_one in j["restriction"]["()"]:
							event["restriction"]["()"].append(in_one)

				if "agent" in event and event["agent"]["entity"] != "":
					event["agent"]["entity"] = event["agent"]["entity"][0:-1]
				if "operation" in event and event["operation"]["operation"] != "":
					event["operation"]["operation"] = event["operation"]["operation"][0:-1]

			if "operation" in event and event["operation"]["operation"] != "":
				tag_list_operation = deal_operation(op_sent, event["operation"]["operation"])
			if "agent" in event and event["agent"]["entity"] != "":
				tag_list_agent = deal_agent(one_sent, event["agent"]["entity"])
			if "input" in event and event["input"]["()"] != []:
				tag_list_input = deal_input(one_sent, event["input"]["()"])
			if "output" in event and event["output"]["()"] != []:
				tag_list_output = deal_output(one_sent, event["output"]["()"])
			if "restriction" in event and event["restriction"]["()"] != []:
				tag_list_restriction = deal_restriction(one_sent, event["restriction"]["()"])
			# 为当前的sentence生成结果列表，加上各自的标注特征，保证各个列表的token是一致的

			for q, word in enumerate(sent.words):
				if word.text.lower() == "then" or word.text.lower() == "only":
					restrictionTag = 1
				else:
					restrictionTag = 0
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




