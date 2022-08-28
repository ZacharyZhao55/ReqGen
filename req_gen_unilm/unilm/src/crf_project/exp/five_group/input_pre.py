import json, stanza, re

nlp = stanza.Pipeline('en', processors='tokenize, pos, lemma, depparse')

open_list = ['E', 'A', 'B', 'C', 'D']
file_path = '/Users/tuanz_lu/PycharmProjects/Pytorch-learning'

for open_tag in open_list:
	res = []

	file = open(file_path+'/json_data/five/'+open_tag+'_data.json')
	fileJson = json.load(file)
	flag = 0
	requirements = []
	inputList = []
	inputList_cur = []
	for i, one in enumerate(fileJson):
		if flag > i:
			inputList_cur = []
			continue
		else:
			line = one[":"]
			requirements.append(line)
			flag = i + 1
			if "input" in one:
				inputList_cur = one["input"]["()"]
			while flag < len(fileJson) and fileJson[flag][":"] == line:  # 合并同一个需求的operation
				if "input" in fileJson[flag]:
					for p, test_one in enumerate(fileJson[flag]["input"]["()"]):
						inputList_cur.append(test_one)
				flag += 1
			inputList.append(inputList_cur)

	operation_file = open(file_path+'/exp/pre/operation_pre_'+open_tag+'.txt')
	rl = operation_file.readlines()
	pos = 0
	for i, one in enumerate(requirements):
		doc = nlp(one)
		for sent in doc.sentences:
			words = sent.words
			one_sent = [i.text.lower() for i in words]  # 根据stanza分出来的words拼成sentence的列表
			tag_list = ["O" for i in range(len(one_sent))]  # 标记列表
			for inp in inputList[i]:
				inp["entity"] = inp["entity"].replace(",", " ,")
				one_inp = inp["entity"].lower().split()  # 每个input的分词列表
				results = [i for i, word in enumerate(one_inp) if re.search('/', word)]
				if results != []:
					one_inp_sent = ' '.join(one_inp)
					p_os = one_inp_sent.find('/')
					tmp = list(one_inp_sent)
					tmp.insert(p_os, ' ')
					tmp.insert(p_os + 2, ' ')
					one_inp_sent = ''.join(tmp)
					one_inp = one_inp_sent.split()
				if "uav's" in one_inp:
					position = one_inp.index("uav's")
					one_inp[position] = "'s"
					one_inp.insert(position, 'uav')
				ll = len(one_inp)
				for j in range(len(one_sent)):
					if "shunting" in one_inp:
						print(one_inp)
						print(one_sent[j:j + ll])
					if one_sent[j:j + ll] == one_inp:
						tag_list[j] = "B"
						if ll != 1:
							tag_list[j + ll - 1] = "E"
						m = j + 1
						while m < j + ll - 1:
							tag_list[m] = "I"
							m += 1
						break

			for q, word in enumerate(sent.words):
				opLine = rl[pos].split()
				opTag = opLine[-1]
				pos += 1
				res.append(f'{word.text}\t{word.xpos}\t{word.deprel}\t{opTag}\t{tag_list[q]}\n')
		res.append(" \n")
		pos += 1

	fw = open(file_path+'/exp/pre/input_pre_'+open_tag+'.txt', 'w')
	for r in res:
		fw.write(r)
	fw.close()


