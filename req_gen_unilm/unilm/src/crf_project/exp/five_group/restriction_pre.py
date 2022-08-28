import json, stanza

nlp = stanza.Pipeline('en', processors='tokenize, pos, lemma, depparse')

open_list = ['E', 'A', 'B', 'C', 'D']
file_path = '/Users/tuanz_lu/PycharmProjects/Pytorch-learning'

restriction_num = 0

for open_tag in open_list:
	res = []

	file = open(file_path+'/json_data/five/'+open_tag + '_data.json')
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
			if "restriction" in one:
				restriction_num += 1
				inputList_cur = one["restriction"]["()"]
			flag = i + 1
			while flag < len(fileJson) and fileJson[flag][":"] == line:  # 合并同一个需求的operation
				if "restriction" in fileJson[flag]:
					for p, test_one in enumerate(fileJson[flag]["restriction"]["()"]):
						inputList_cur.append(test_one)
				flag += 1
			inputList.append(inputList_cur)

	for i, one in enumerate(requirements):
		doc = nlp(one)
		for sent in doc.sentences:
			words = sent.words
			one_sent = [i.text.lower() for i in words]  # 根据stanza分出来的words拼成sentence的列表
			tag_list = ["O" for i in range(len(one_sent))]  # 标记列表
			for inp in inputList[i]:
				one_inp = inp.lower().split()  # 每个output的分词列表
				ll = len(one_inp)
				for j in range(len(one_sent)):
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
				if word.text.lower() == "then" or word.text.lower() == "only":
					opTag = 1
				else:
					opTag = 0
				res.append(f'{word.text}\t{word.xpos}\t{word.deprel}\t{opTag}\t{tag_list[q]}\n')
		res.append(" \n")

	fw = open(file_path+'/exp/pre/restriction_pre_'+open_tag + '.txt', 'w')
	for r in res:
		fw.write(r)
	fw.close()

