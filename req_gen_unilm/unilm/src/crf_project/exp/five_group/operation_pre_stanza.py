import json
import stanza
nlp = stanza.Pipeline('en', processors='tokenize, pos, lemma, depparse')
open_list = ['E', 'A', 'B', 'C', 'D']
file_path = '/Users/tuanz_lu/PycharmProjects/Pytorch-learning'
conditions = ["if", "when"]

for open_tag in open_list:
	res = []
	file = open(file_path+'/json_data/five/'+open_tag+'_data.json')
	fileJson = json.load(file)
	flag = 0
	operation = ""
	whether_main = 1
	requirements = []
	for i, one in enumerate(fileJson):
		if flag > i:
			operation = ""  # 对新的需求operation进行初始化
			continue
		else:
			line = one[":"]
			operation = one["operation"]["operation"]
			flag = i + 1
			while flag < len(fileJson) and fileJson[flag][":"] == line:  # 合并同一个需求的operation
				operation += " " + fileJson[flag]["operation"]["operation"]
				flag += 1
			ops = operation.split()
			doc = nlp(line)
			for j, sent in enumerate(doc.sentences):
				for word in sent.words:
					if word.lemma in conditions:
						whether_main = 0 # 0代表主句， 1代表从句
					if len(ops) > 0 and word.lemma == ops[0] and whether_main == 1:
						tag = 1
						ops = ops[1:]
					else:
						tag = 0
					if whether_main == 0 and (word.lemma == "," or word.lemma == "."):
						whether_main = 1
					res.append(f'{word.text}\t{word.xpos}\t{word.deprel}\t{tag}\n')
			res.append(" \n")

	fw = open(file_path+'/exp/pre/operation_pre_'+open_tag+'.txt', 'w')
	for r in res:
		fw.write(r)
	fw.close()
