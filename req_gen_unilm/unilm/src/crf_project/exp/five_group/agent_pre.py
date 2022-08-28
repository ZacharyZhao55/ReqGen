import json
import stanza

nlp = stanza.Pipeline('en', processors='tokenize, pos, lemma, depparse')
open_list = ['E', 'A', 'B', 'C', 'D']
conditions = ["if", "when", "then", "only"]
file_path = '/Users/tuanz_lu/PycharmProjects/Pytorch-learning'

for open_tag in open_list:
	res = []
	file = open(file_path+'/json_data/five/'+open_tag+'_data.json')
	fileJson = json.load(file)

	flag = 0
	whether_main = 1
	operation = ""
	for i, one in enumerate(fileJson):
		if flag > i:
			print(flag)
			operation = ""  # 对新的需求operation进行初始化
			continue
		else:
			line = one[":"]
			ops = []
			pos = -1
			flag = i + 1
			while flag < len(fileJson) and fileJson[flag][":"] == line:  # 合并同一个需求的operation
				flag += 1
			if "agent" in one:
				agent = one["agent"]["entity"]
				if '/' in agent:
					pos = agent.find('/')
					tmp = list(agent)
					tmp.insert(pos, ' ')
					tmp.insert(pos + 2, ' ')
					agent = ''.join(tmp)
				ops = agent.split()
			doc = nlp(line)
			ll = len(ops)
			for j, sent in enumerate(doc.sentences):

				for word in sent.words:
					if word.lemma in conditions:  # 划分从句
						whether_main = 0 # 0代表主句， 1代表从句
					if len(ops) > 0 and word.text.lower() == ops[0].lower() and whether_main == 1:

						if ll == len(ops):
							tag = "B"
						elif ll != 1 and len(ops) == 1:
							tag = "E"
						else:
							tag = "I"
						ops = ops[1:]
					else:
						tag = "O"
					if whether_main == 0 and (word.lemma == "," or word.lemma == "."):
						whether_main = 1
					if word.text == "_UAVActivation" or word.text == "_Activity" or word.text == "_UAVRegistration":
						tag = "B"
					elif word.text == "Manager_" or word.text == "Logger_":
						tag = "E"
					res.append(f'{word.text}\t{word.xpos}\t{word.deprel}\t{tag}')
			res.append(" \n")

	operation_file = open(file_path+'/exp/pre/operation_pre_'+open_tag+'.txt')
	rl = operation_file.readlines()
	for i, line in enumerate(res):
		lline = line.split()
		if len(lline) > 3:
			tagg = lline[-1]
			rl[i] = rl[i][0:-1]
			rl[i] += f"\t{tagg}\n"
	fw = open(file_path+'/exp/pre/agent_pre_'+open_tag+'.txt', 'w')
	for r in rl:
		fw.write(r)
	fw.close()


