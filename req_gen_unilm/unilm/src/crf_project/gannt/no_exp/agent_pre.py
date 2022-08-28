import stanza

nlp = stanza.Pipeline('en', processors='tokenize, pos, lemma, depparse')
# open_list = ['uav', 'pure', 'gannt', 'worldvista']
open_list = ['gannt']
conditions = ["if", "when", "then", "only"]
# 此处file_path对应structure_path
file_path = '/root/Desktop/crf_project/crf_project'

for open_tag in open_list:
	res = []

	# 此处打开需求集
	file = open(file_path+'/json_data/'+open_tag+'_data.txt')
	fileJson = file.readlines()

	flag = 0
	whether_main = 1
	operation = ""
	for i, one in enumerate(fileJson):
		line = one[0:-1]
		doc = nlp(line)
		for j, sent in enumerate(doc.sentences):
			for word in sent.words:
				tag = "O"
				res.append(f'{word.text}\t{word.xpos}\t{word.deprel}\t{tag}')
		res.append(" \n")

	# 此处打开operation的预处理文件
	operation_file = open(file_path+'/exp/pre/operation_pre_'+open_tag+'.txt')
	rl = operation_file.readlines()
	for i, line in enumerate(res):
		lline = line.split()
		if len(lline) > 3:
			tagg = lline[-1]
			rl[i] = rl[i][0:-1]
			rl[i] += f"\t{tagg}\n"

	# 此处写入agent的预处理文件
	fw = open(file_path+'/exp/pre/agent_pre_'+open_tag+'.txt', 'w')
	for r in rl:
		fw.write(r)
	fw.close()


print("agent pre done")