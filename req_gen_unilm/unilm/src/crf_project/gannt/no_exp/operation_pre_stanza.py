import stanza
nlp = stanza.Pipeline('en', processors='tokenize, pos, lemma, depparse')
# open_list = ['uav', 'pure', 'gannt', 'worldvista']
open_list = ['gannt']


file_path = '/root/Desktop/crf_project/crf_project'

conditions = ["if", "when"]

for open_tag in open_list:
	res = []
	# 此处打开需求集
	file = open(file_path+'/json_data/'+open_tag+'_data.txt')
	fileJson = file.readlines()
	whether_main = 1
	for i, one in enumerate(fileJson):
		line = one[0:-1]
		doc = nlp(line)
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
	fw = open(file_path+'/exp/pre/operation_pre_'+open_tag+'.txt', 'w')
	for r in res:
		fw.write(r)
	fw.close()


print("operation pre done")