import json
import pandas as pd
import os


# gensim
import gensim
from gensim.test.utils import common_texts
from gensim.models.doc2vec import TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]


def event_cp(file_data, output_path):

	word_list = ["driver", "train", "uav"]

	result_list = []

	# 计算event短句与其他需求的句子相似性
	for i, item_one in enumerate(file_data):
		for j, item_two in enumerate(file_data):
			if item_one[":"] == item_two[":"]:
				continue
			if "event" in item_one:
				sent_two = item_two[":"]
				if "event" in item_two:
					sent_two = sent_two.replace(item_two["event"][0][":"].lower(), "")
				if similarity_test(item_one["event"][0][":"], sent_two) < 0.328:
					result_item = {
						"req1": item_one[":"],
						"req1_id": item_one["#"],
						"req2": item_two[":"],
						"req2_id": item_two["#"],
						"req1_event": item_one["event"][0][":"],
						"req2_main": sent_two,
						"similarity": similarity_test(item_one["event"][0][":"], sent_two),
						"rel": 1,
						"tag_rel": 1
					}
					result_list.append(result_item)
	with open(output_path, 'w') as f:
		json.dump(result_list, f)

def input_output_cp(file_data, output_path):

	result_list = []

	word_list = ["driver", "train", "uav", "ability","public method"]
	# 用wmd距离判断
	for i, item_one in enumerate(file_data):
		for j, item_two in enumerate(file_data):
			if i>j or item_one[":"] == item_two[":"]:
				continue
			flag = 0
			if item_one["input"] != [] and item_two["output"] != []:
				for one_input in item_one["input"]:
					for one_output in item_two["output"]:
						if (one_input == one_output or similarity_test(one_input, one_output) < 0.3) and one_input not in word_list:
							if (("new" in one_input and "original" in one_output) or (
									"new" in one_output and "original" in one_input)):
								continue
							if (("task" in one_input and "holiday" in one_output) or (
									"task" in one_output and "holiday" in one_input)):
								continue
							if (("human" in one_input and "two tasks" in one_output) or (
									"human" in one_output and "two tasks" in one_input)):
								continue
							result_item = {
								"req1": item_one,
								"req2": item_two,
								"rel":"3-1"
							}
							result_list.append(result_item)
							flag = 1
							break

					if flag == 1:
						break
			if flag == 1:
				continue
			else:
				if item_one["output"] != [] and item_two["input"] != []:
					for one_input in item_two["input"]:
						for one_output in item_one["output"]:
							if (one_input == one_output or similarity_test(one_input, one_output) < 0.3) and one_input not in word_list:
								if (("new" in one_input and "original" in one_output) or ("new" in one_output and "original" in one_input)):
									continue
								if (("task" in one_input and "holiday" in one_output) or ("task" in one_output and "holiday" in one_input)):
									continue
								if (("human" in one_input and "two tasks" in one_output) or (
										"human" in one_output and "two tasks" in one_input)):
									continue

								result_item = {
									"req1": item_one,
									"req2": item_two,
									"rel": "3-2"
								}
								result_list.append(result_item)
								flag = 1
								break
						if flag == 1:
							break

	with open(output_path, 'w') as f:
		json.dump(result_list, f)

def similarity_test(s1, s2):
	# corpus = api.load('text8')  # download the corpus and return it opened as an iterable

	# sentences = word2vec.Text8Corpus('./model/text8')
	# model = Word2Vec(sentences)  # train a model from the corpus
	# save_model_name = 'text8'
	# save_model_file = 'text8.model'
	# model.save(save_model_file)
	# model.wv.save_word2vec_format(save_model_name + ".bin", binary=True)

	# model = word2vec.Word2Vec.load(save_model_file)
	# y1 = model.wv.similarity("sentence", "question")
	# print(u"新闻和热度的相似度为：", y1)
	# print("-------------------------------\n")

	model = gensim.models.KeyedVectors.load_word2vec_format('./text8.bin', binary=True)
	distance = model.wmdistance(s1, s2)
	return distance



def compare_input_result(input_path, csv_path):

	result_data = json.load(open(input_path))
	file_data = pd.read_csv(csv_path)

	total_num = 0
	correct_num = 0
	include_num = 0
	non_correct_num = 0

	for index, row in file_data.iterrows():
		if row["label"] == 3.0:
			total_num += 1

	for i, item in enumerate(result_data):
		req1 = item["req1"][":"]
		req2 = item["req2"][":"]
		for index, row in file_data.iterrows():

			if ((row["req1"].lower() in req1 or req1 == row["req1"].lower()) and (row["req2"].lower() in req2 or req2 == row["req2"].lower()) ) or ((row["req1"].lower() in req2 or req2 == row["req1"].lower())  and (row["req2"].lower() in req1 or req1 == row["req2"].lower()) ):
				if row["label"] == 3:
					correct_num += 1
				else:
					non_correct_num += 1
				include_num += 1
				print(item["req1"])
				print(item["req2"])
				print(row["label"])
				print()
				break


	print(len(result_data))
	print("include_num : "+ str(include_num))

	print("correct_num : "+ str(correct_num))
	print("total_num : "+ str(total_num))
	Pacc = correct_num / include_num
	Racc = correct_num / total_num
	if Pacc + Racc != 0:
		Fmeasure = (2 * Pacc * Racc) / (Pacc + Racc)
	else:
		Fmeasure = 0
	print("P-accuracy : " + str(Pacc))
	print("R-accuracy: " + str(Racc))
	print("F-measure: " + str(Fmeasure))


def compare_event_result(input_path, csv_path):

	event_data = json.load(open(input_path))
	file_data = pd.read_csv(csv_path)

	total_num = 0
	correct_num = 0
	include_num = 0
	data_cnt = [0 for i in range(len(file_data))]
	for index, row in file_data.iterrows():
		if row["label"] in [1, 2]:
			total_num += 1

	for i, item in enumerate(event_data):
		req1 = item["req1"]
		req2 = item["req2"]
		for index, row in file_data.iterrows():
			if (req1 == row["req1"].lower() or row["req1"].lower() in req1) and (req2 == row["req2"].lower() or row["req2"].lower() in req2) and data_cnt[index] == 0:
				if row["label"] in [1,2]:
					correct_num += 1
					data_cnt[index] = 1
				include_num += 1
				print(item["req1"])
				print(item["req2"])
				print(row["label"])
				print()
				break
			elif (req2 == row["req1"].lower() or row["req1"].lower() in req2) and (req1 == row["req2"].lower() or row["req2"].lower() in req1) and data_cnt[index] == 0:
				if row["label"] in [1,2]:
					correct_num += 1
					data_cnt[index] = 1
				include_num += 1
				print(item["req1"])
				print(item["req2"])
				print(row["label"])
				print()
				break


	print(len(event_data))
	print("include_num : "+ str(include_num))
	print("correct_num : "+ str(correct_num))
	print("total_num : "+ str(total_num))

	Pacc = correct_num/include_num
	Racc = correct_num/total_num
	if Pacc+Racc != 0:
		Fmeasure = (2*Pacc*Racc)/(Pacc+Racc)
	else:
		Fmeasure = 0
	print("P-accuracy : " + str(Pacc))
	print("R-accuracy: " + str(Racc))
	print("F-measure: " + str(Fmeasure))


def input_output_relation(file_data, output_path):

	result_list = []

	word_list = ["driver", "train", "uav", "ability"]
	# 用wmd距离判断

	for i, item_one in enumerate(file_data):
		e_flag = 0
		edge_list = []
		for j, item_two in enumerate(file_data):
			if i>j or item_one[":"] == item_two[":"]:
				continue
			flag = 0

			if item_one["input"] != [] and item_two["output"] != []:
				for one_input in item_one["input"]:
					for one_output in item_two["output"]:
						if (similarity_test(one_input, one_output) < 0.2 or one_input == one_output) and one_input not in word_list:
							edge_item = {
								"input": item_one["#"],
								"input_req": item_one[":"],
								"output": item_two["#"],
								"output_req": item_two[":"]
							}
							e_flag = 1
							edge_list.append(edge_item)
							flag = 1
							break

					if flag == 1:
						break
		if e_flag == 1:
			node_item = {
				"node_id": item_one["#"],
				"node_req": item_one[":"],
				"edges": edge_list
			}
			result_list.append(node_item)

	with open(output_path, 'w') as f:
		json.dump(result_list, f)


root_path="/Users/tuanz_lu/PycharmProjects/Pytorch-learning"

gannt_file_path = root_path+'/gannt/back/test_gannt.json'
pure_file_path = root_path+'/gannt/back/test_pure.json'

pure_file_data = json.load(open(pure_file_path))
gannt_file_data = json.load(open(gannt_file_path))



# gannt
# if not os.path.exists(root_path+'/priority/gannt'):
# 	os.makedirs(root_path+'/priority/gannt')
event_cp(gannt_file_data, root_path+'/priority/gannt/wmd_event.json')
# input_output_cp(gannt_file_data, root_path+
# 				'/priority/gannt/wmd_interaction.json')
# input_output_relation(gannt_file_data, root_path+'/priority/gannt/wmd_input_to_output.json')
# compare_input_result(root_path+'/priority/gannt/wmd_interaction.json', root_path+'/priority/gannt_cos.csv')
# compare_event_result(root_path+'/priority/gannt/wmd_event.json', root_path+'/priority/gannt_cos.csv')


# pure
# if not os.path.exists(root_path+'/priority/pure'):
# 	os.makedirs(root_path+'/priority/pure')
# event_cp(pure_file_data, root_path+'/priority/pure/wmd_event.json')
# input_output_cp(pure_file_data, root_path+'/priority/pure/wmd_interaction.json')
# input_output_relation(pure_file_data, root_path+'/priority/pure/wmd_input_to_output.json')
# compare_input_result(root_path+'/priority/pure/wmd_interaction.json', root_path+'/priority/pure_unified.csv')
# compare_event_result(root_path+'/priority/pure/wmd_event.json', root_path+'/priority/pure_unified.csv')
#