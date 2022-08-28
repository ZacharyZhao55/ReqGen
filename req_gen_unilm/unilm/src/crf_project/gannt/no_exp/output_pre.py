import stanza

nlp = stanza.Pipeline('en', processors='tokenize, pos, lemma, depparse')
# open_list = ['uav', 'pure', 'gannt', 'worldvista']
open_list = ['gannt']



file_path = '/root/Desktop/crf_project/crf_project'

output_num = 0
for open_tag in open_list:
	res = []
	# 此处打开需求集
	file = open(file_path+'/json_data/'+open_tag+'_data.txt')
	fileJson = file.readlines()
	requirements = []
	for i, one in enumerate(fileJson):
		line = one[0:-1]
		requirements.append(line)

	# 此处打开operation预处理文件
	operation_file = open('../exp/pre/operation_pre_'+open_tag+'.txt')
	rl = operation_file.readlines()
	pos = 0
	# 读入operation_pre.txt 加入operation字段
	for i, one in enumerate(requirements):
		doc = nlp(one)
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
	fw = open(file_path+'/exp/pre/output_pre_'+open_tag+'.txt', 'w')
	for r in res:
		fw.write(r)
	fw.close()


print("output pre done")