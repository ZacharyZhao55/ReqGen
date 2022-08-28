import json, stanza, random, os

# file_path为项目根目录地址
file_path = '/Users/tuanz_lu/PycharmProjects/Pytorch-learning'

file_uav = open(file_path+'/json_data/UAV.json')
file_gannt = open(file_path+'/json_data/GANNT.json')
file_pure = open(file_path+'/json_data/PURE.json')
file_worldvista = open(file_path+'/json_data/WORLDVISTA.json')

uav_data = json.load(file_uav)
gannt_data = json.load(file_gannt)
pure_data = json.load(file_pure)
worldvista_data = json.load(file_worldvista)
nlp = stanza.Pipeline('en', processors='tokenize, pos, lemma, depparse')

def deal_req(sent):

	# 预处理 for operation
	sent = sent.replace("provide functionality to ", "")
	sent = sent.replace("provide means to ", "")
	sent = sent.replace("provide the ability to ", "")
	sent = sent.replace("provide a means to ", "")
	sent = sent.replace("needs to provide", "provides")
	sent = sent.replace("should able to provide", "provides")
	sent = sent.replace("should be able to provide", "provides")
	sent = sent.replace("provides functions to", "shall")
	sent = sent.replace("provides interface and functions to", "shall")
	sent = sent.replace("provides a function to", "shall")
	sent = sent.replace("provides ability to", "shall")
	sent = sent.replace("provides an ability to", "shall")
	sent = sent.replace("provides methods to", "shall")
	sent = sent.replace("provides a method to", "shall")
	sent = sent.replace("provide a way to", "")
	sent = sent.replace("have the ability to", "")
	sent = sent.replace("have the capacity to", "")
	sent = sent.replace("need to", "shall")
	sent = sent.replace("needs to", "shall")
	if "takes care of" in sent and "handling" in sent:
		sent = sent.replace("handling", "handle")
		sent = sent.replace("releasing", "release")
		sent = sent.replace("takes care of", "shall")

	if "have the capability of" in sent:

		doc = nlp(sent)
		for one in doc.sentences:
			for i, word in enumerate(one.words):
				if word.xpos == "VBG" and (word.deprel == 'advcl' or word.deprel == 'aux:pass'):
					sent = sent.replace(word.text, word.lemma)
		sent = sent.replace("have the capability of", "shall")

	# if "be capable of" in sent:
	# 	doc = nlp(sent)

	# 	for one in doc.sentences:
	# 		for i, word in enumerate(one.words):
	# 			if word.xpos == "VBG" and (word.deprel == 'advcl' or word.deprel == 'aux:pass'):
	# 				sent = sent.replace(word.text, word.lemma)
	# 	sent = sent.replace("be capable of ", "")


	# for event
	sent = sent.replace("In case", "If")
	sent = sent.replace("as long as", "if")
	sent = sent.replace("while", "when")
	sent = sent.replace("Where", "When")
	sent = sent.replace("Every time", "When")
	sent = sent.replace("Once", "When")
	sent = sent.replace("each time", "when")


	return sent


def get_pass(data):
	conditions = ["if", "when"]

	print("in get_pass")
	print(len(data))
	posi_list = []
	pass_list = []
	flag = 0
	whether_main = 1
	for i, one in enumerate(data):
		if flag > i:
			continue
		else:
			line = one[":"]
			flag = i + 1
			while flag < len(data) and data[flag][":"] == line:  # 合并同一个需求的operation
				flag += 1
			doc = nlp(line)
			seperate_flag = 0
			# print(line)
			for j, sent in enumerate(doc.sentences):

				for word in sent.words:
					if word.lemma in conditions:  # 划分从句
						whether_main = 0  # 0代表从句， 1代表主句
					if whether_main == 1:
						if seperate_flag == 0 and "pass" in word.deprel:
							seperate_flag = 1
							break
					if whether_main == 0 and (word.lemma == "," or word.lemma == "."):
						whether_main = 1
				if seperate_flag == 1:
					break
		if seperate_flag == 0:
			posi_list.append(one)
		else:
			pass_list.append(one)
	random.shuffle(posi_list)
	random.shuffle(pass_list)
	return posi_list, pass_list

def get_five_group(data, outputPath):

	A_list = []
	B_list = []
	C_list = []
	D_list = []
	E_list = []
	flag = 0
	#划分主句语态
	posi_data, pass_data = get_pass(data)
	posi_len = int(len(posi_data)/5)
	pass_len = int(len(pass_data)/5)
	print('posi_len:' + str(posi_len))
	print('pass_len:' + str(pass_len))
	# 分配主动数据
	for i, one in enumerate(posi_data):
		if flag > i:
			continue
		else:
			line = one[":"]
			flag = i + 1
			while flag < len(posi_data) and posi_data[flag][":"] == line:
				one["operation"]["operation"] += " " + posi_data[flag]["operation"]["operation"]
				if "input" in one and "input" in posi_data[flag]:
					one["input"]["()"] += posi_data[flag]["input"]["()"]
				elif not "input" in one and "input" in posi_data[flag]:
					one["input"] = {
						"()": posi_data[flag]["input"]["()"]
					}

				if "output" in one and "output" in posi_data[flag]:
					one["output"]["()"] += posi_data[flag]["output"]["()"]
				elif not "output" in one and "output" in posi_data[flag]:
					one["output"] = {
						"()": posi_data[flag]["output"]["()"]
					}

				if "restriction" in one and "restriction" in posi_data[flag]:
					one["restriction"]["()"] += posi_data[flag]["restriction"]["()"]
				elif not "restriction" in one and "restriction" in posi_data[flag]:
					one["restriction"] = {
						"()": posi_data[flag]["restriction"]["()"]
					}

				flag += 1
			new_list = []

			new_item = {
				"before": one[":"],
				"after": deal_req(one[":"])
			}
			# 处理requirements中的provide 语句
			one[":"] = new_item["after"]
			# print(one[":"])

			if "input" in one:
				for in_item in one["input"]["()"]:
					if not in_item in new_list:
						new_list.append(in_item)
				one["input"]["()"] = new_list
				new_list = []

			if "output" in one:
				for in_item in one["output"]["()"]:
					if not in_item in new_list:
						new_list.append(in_item)
				one["output"]["()"] = new_list
				new_list = []

			if "restriction" in one:
				for in_item in one["restriction"]["()"]:
					if not in_item in new_list:
						new_list.append(in_item)
				one["restriction"]["()"] = new_list
				new_list = []

			if len(A_list) < posi_len:
				A_list.append(one)
			elif len(A_list) == posi_len and len(B_list) < posi_len:
				B_list.append(one)
			elif len(B_list) == posi_len and len(C_list) < posi_len:
				C_list.append(one)
			elif len(C_list) == posi_len and len(D_list) < posi_len:
				D_list.append(one)
			elif len(D_list) == posi_len:
				E_list.append(one)

	flag = 0
	# 分配被动数据
	for i, one in enumerate(pass_data):
		if flag > i:
			print('here got repeat pass')
			print(one)
			continue
		else:
			line = one[":"]
			flag = i + 1
			while flag < len(pass_data) and pass_data[flag][":"] == line:
				one["operation"]["operation"] += " " + pass_data[flag]["operation"]["operation"]
				if "input" in one and "input" in pass_data[flag]:
					one["input"]["()"] += pass_data[flag]["input"]["()"]
				elif not "input" in one and "input" in pass_data[flag]:
					one["input"] = {
						"()": pass_data[flag]["input"]["()"]
					}

				if "output" in one and "output" in pass_data[flag]:
					one["output"]["()"] += pass_data[flag]["output"]["()"]
				elif not "output" in one and "output" in pass_data[flag]:
					one["output"] = {
						"()": pass_data[flag]["output"]["()"]
					}

				if "restriction" in one and "restriction" in pass_data[flag]:
					one["restriction"]["()"] += pass_data[flag]["restriction"]["()"]
				elif not "restriction" in one and "restriction" in pass_data[flag]:
					one["restriction"] = {
						"()": pass_data[flag]["restriction"]["()"]
					}

				flag += 1
			new_list = []

			new_item = {
				"before": one[":"],
				"after": deal_req(one[":"])
			}
			# 处理requirements中的provide 语句
			one[":"] = new_item["after"]
			# print(one[":"])

			if "input" in one:
				for in_item in one["input"]["()"]:
					if not in_item in new_list:
						new_list.append(in_item)
				one["input"]["()"] = new_list
				new_list = []

			if "output" in one:
				for in_item in one["output"]["()"]:
					if not in_item in new_list:
						new_list.append(in_item)
				one["output"]["()"] = new_list
				new_list = []

			if "restriction" in one:
				for in_item in one["restriction"]["()"]:
					if not in_item in new_list:
						new_list.append(in_item)
				one["restriction"]["()"] = new_list
				new_list = []

			if len(A_list) < posi_len + pass_len:
				A_list.append(one)
				# print('A got pass')
			elif len(A_list) == posi_len + pass_len and len(B_list) < posi_len + pass_len:
				B_list.append(one)
				# print('B got pass')

			elif len(B_list) == posi_len + pass_len and len(C_list) < posi_len + pass_len:
				C_list.append(one)
				# print('C got pass')

			elif len(C_list) == posi_len + pass_len and len(D_list) < posi_len + pass_len:
				D_list.append(one)
				# print('D got pass')

			elif len(D_list) == posi_len + pass_len:
				E_list.append(one)
				# print('E got pass')


	open_list = ['A', 'B', 'C', 'D', 'E']
	output_list = [A_list, B_list, C_list, D_list, E_list]
	for i, tag in enumerate(open_list):
		if not os.path.exists(outputPath):
			os.makedirs(outputPath)
		with open(outputPath+tag+'_data.json', 'w') as f:
			print(len(output_list[i]))
			for j, one in enumerate(output_list[i]):
				one["#"] = j+1
			json.dump(output_list[i], f)


get_five_group(uav_data+pure_data+gannt_data+worldvista_data, file_path+'/json_data/five/')