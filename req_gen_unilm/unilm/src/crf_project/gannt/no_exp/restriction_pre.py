import stanza

nlp = stanza.Pipeline('en', processors='tokenize, pos, lemma, depparse')
# open_list = ['uav', 'pure', 'gannt', 'worldvista']
open_list = ['gannt']



file_path = '/root/Desktop/crf_project/crf_project'

restriction_num = 0

for open_tag in open_list:
	res = []

	# 此处打开需求集
	file = open(file_path+'/json_data/'+open_tag + '_data.txt')
	fileJson = file.readlines()
	requirements = []
	for i, one in enumerate(fileJson):
		line = one[0:-1]
		requirements.append(line)

	for i, one in enumerate(requirements):
		doc = nlp(one)
		for sent in doc.sentences:
			words = sent.words
			one_sent = [i.text.lower() for i in words]  # 根据stanza分出来的words拼成sentence的列表
			tag_list = ["O" for i in range(len(one_sent))]  # 标记列表
			for q, word in enumerate(sent.words):
				if word.text.lower() == "then" or word.text.lower() == "only":
					opTag = 1
				else:
					opTag = 0
				res.append(f'{word.text}\t{word.xpos}\t{word.deprel}\t{opTag}\t{tag_list[q]}\n')
		res.append(" \n")

	# 此处写入restriction预处理文件
	fw = open(file_path+'/exp/pre/restriction_pre_'+open_tag + '.txt', 'w')
	for r in res:
		fw.write(r)
	fw.close()

print("restriction pre done")