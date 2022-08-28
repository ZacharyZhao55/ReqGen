import json
from curses.ascii import isalnum
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma, depparse')

special_agent_words = ["_UAVRegistration", '_UAVActivation', '_Activity']
file_path = '/Users/tuanz_lu/PycharmProjects/Pytorch-learning/exp'
final_list = []

def get_agent(inPath, outPath):
	f = open(inPath)
	rl = f.readlines()
	res = []
	item = {
		"#": "",
		":": "",
		"agent": ""
	}

	for i, line in enumerate(rl):
		if line != "\n":
			line_words = line.split()
			line_words[0] = line_words[0].lower()
			if len(line_words[0]) == 1 and not isalnum(line_words[0]):
				item[":"] = item[":"][0:-1]

			if line_words[0][-1] == '/':
				item[":"] += line_words[0]
			else:
				item[":"] += line_words[0] + " "
			if line_words[5] == "B" or line_words[5] == "I" or line_words[5] == "E":
				mini_special = [oo.lower() for oo in special_agent_words]
				if line_words[0] in mini_special:
					item["agent"] += line_words[0]
				elif line_words[0] == '/' or line_words[0] == '-':
					item["agent"] = item["agent"][0:-1] + line_words[0]
				else:
					item["agent"] += line_words[0] + " "
		else:
			item["#"] += str(len(res) + 1)
			item[":"] = item[":"][0:-1]
			if len(item["agent"]) > 1:
				item["agent"] = item["agent"][0:-1]
			res.append(item)
			item = {
				"#": "",
				":": "",
				"agent": ""
			}

	with open(outPath, 'w') as fo:
		json.dump(res, fo)


def get_operation(inPath, outPath):
	res = json.load(open(outPath))
	rl = open(inPath).readlines()
	for i in res:
		i['operation'] = ''
	cur = 0
	for i, line in enumerate(rl):
		if line != "\n":
			line_words = line.split()
			line_words[0] = line_words[0].lower()
			if line_words[4] == '1':
				res[cur]["operation"] += line_words[0] + " "
		else:
			if len(res[cur]["operation"]) > 1:
				res[cur]["operation"] = res[cur]["operation"][0:-1]
			cur += 1
	with open(outPath, 'w') as f:
		json.dump(res, f)


def get_input(inPath, outPath):
	res = json.load(open(outPath))
	rl = open(inPath).readlines()
	for i, one in enumerate(res):
		one['input'] = []
	cur = 0
	cur_input = ""
	for i, line in enumerate(rl):
		if line != "\n":
			line_words = line.split()
			line_words[0] = line_words[0].lower()
			if line_words[5] == 'B':
				cur_input += line_words[0] + " "
			elif line_words[5] == "I":
				if line_words[0] == '/':
					cur_input = cur_input[0:-1]
					cur_input += line_words[0]
				elif line_words[0] == "'s":
					cur_input = cur_input[0:-1]
					cur_input += line_words[0] + " "
				else:
					cur_input += line_words[0] + " "
			elif line_words[5] == "E":
				cur_input += line_words[0] + " "
				res[cur]["input"].append(cur_input[0:-1])
				cur_input = ""
			elif line_words[5] == "O":
				if cur_input != "":
					res[cur]["input"].append(cur_input[0:-1])
					cur_input = ""
		else:
			cur += 1
	with open(outPath, 'w') as f:
		json.dump(res, f)


def get_output(inPath, outPath):
	res = json.load(open(outPath))
	rl = open(inPath).readlines()
	for i in res:
		i['output'] = []
	cur = 0
	cur_input = ""
	for i, line in enumerate(rl):
		if line != "\n":
			line_words = line.split()
			line_words[0] = line_words[0].lower()
			if line_words[5] == 'B':
				cur_input += line_words[0] + " "
			elif line_words[5] == "I":
				if line_words[0] == '/':
					cur_input = cur_input[0:-1]
					cur_input += line_words[0]
				else:
					cur_input += line_words[0] + " "
			elif line_words[5] == "E":
				cur_input += line_words[0] + " "
				res[cur]["output"].append(cur_input[0:-1])
				cur_input = ""
			elif line_words[5] == "O":
				if cur_input != "":
					res[cur]["output"].append(cur_input[0:-1])
					cur_input = ""
		else:
			cur += 1
	with open(outPath, 'w') as f:
		json.dump(res, f)


def get_restriction(inPath, outPath):
	res = json.load(open(outPath))
	rl = open(inPath).readlines()
	for i in res:
		i['restriction'] = []
	cur = 0

	cur_input = ""
	for i, line in enumerate(rl):
		if line != "\n":
			line_words = line.split()
			line_words[0] = line_words[0].lower()
			if line_words[5] == 'B':
				if cur_input != "":
					res[cur]["restriction"].append(cur_input[0:-1])
					cur_input = ""
				cur_input += line_words[0] + " "
			elif line_words[5] == "I":
				if line_words[0] == '/':
					cur_input = cur_input[0:-1]
					cur_input += line_words[0]
				else:
					cur_input += line_words[0] + " "
			elif line_words[5] == "E":
				cur_input += line_words[0] + " "
				res[cur]["restriction"].append(cur_input[0:-1])
				cur_input = ""
			elif line_words[5] == "O":
				if cur_input != "":
					res[cur]["restriction"].append(cur_input[0:-1])
					cur_input = ""
		else:
			cur += 1

	with open(outPath, 'w') as f:
		json.dump(res, f)


def get_event(agentPath, operationPath, inputPath, outputPath, restrictionPath, outPath):
	rl = json.load(open(outPath))
	num = 0
	j = 0
	for i, line in enumerate(rl):
		if "when " in line[":"].lower() or "if " in line[":"].lower():
			item = {
				"agent": "",
				"operation": "",
				"input": [],
				"output": [],
				"restriction": [],
				":": ""
			}
			event = []
			# print(" i in if :" + str(i))
			num += 1
			rl_agent = open(agentPath).readlines()
			rl_operation = open(operationPath).readlines()
			rl_input = open(inputPath).readlines()
			rl_output = open(outputPath).readlines()
			rl_restriction = open(restrictionPath).readlines()
			cur_input = ""
			cur_output = ""
			cur_restriction = ""
			while j < len(rl_agent):
				agent_line = rl_agent[j]
				operation_line = rl_operation[j]
				input_line = rl_input[j]
				output_line = rl_output[j]
				restriction_line = rl_restriction[j]

				if agent_line != "\n":
					# agent
					agent_line_words = agent_line.split()
					if len(agent_line_words[0]) == 1 and not isalnum(agent_line_words[0]):
						item[":"] = item[":"][0:-1]
					if agent_line_words[0] == '/' or agent_line_words[0] == '-':
						item[":"] += agent_line_words[0]
					else:
						item[":"] += agent_line_words[0] + " "
					if agent_line_words[5] == "B" or agent_line_words[5] == "I" or agent_line_words[5] == "E":
						if agent_line_words[0] in special_agent_words:
							item["agent"] += agent_line_words[0]
						else:
							item["agent"] += agent_line_words[0] + " "

					# operation
					operation_line_words = operation_line.split()
					if operation_line_words[4] == '1':
						item["operation"] += operation_line_words[0] + " "

					# input
					input_line_words = input_line.split()
					if input_line_words[5] == 'B':
						if cur_input != "":
							item["input"].append(cur_input[0:-1])
							cur_input = ""
						elif j == len(rl_agent)-1 or rl_agent[j+1] == '\n':

							cur_input += input_line_words[0]
							item["input"].append(cur_input)
							cur_input = ""
						else:
							cur_input += input_line_words[0] + " "
					elif input_line_words[5] == "I":
						cur_input += input_line_words[0] + " "
					elif input_line_words[5] == "E":
						cur_input += input_line_words[0] + " "
						cur_input = cur_input.replace(" 's", "'s")
						item["input"].append(cur_input[0:-1])
						cur_input = ""
					elif input_line_words[5] == "O":
						if cur_input != "":
							item["input"].append(cur_input[0:-1])
							cur_input = ""

					# output
					output_line_words = output_line.split()
					if output_line_words[5] == 'B':
						if cur_output != "":
							item["output"].append(cur_output[0:-1])
							cur_output = ""
						elif j == len(rl_agent) - 1 or rl_agent[j+1] == '\n':
							cur_output += input_line_words[0]
							item["output"].append(cur_output)
							cur_output = ""
						else:
							cur_output += input_line_words[0] + " "
					elif output_line_words[5] == "I":
						cur_output += output_line_words[0] + " "
					elif output_line_words[5] == "E":
						cur_output += output_line_words[0] + " "
						cur_output = cur_output.replace(" 's", "'s")
						item["output"].append(cur_output[0:-1])
						cur_output = ""
					elif output_line_words[5] == "O":
						if cur_output != "":
							item["output"].append(cur_output[0:-1])
							cur_output = ""

					# restriction
					restriction_line_words = restriction_line.split()
					if restriction_line_words[5] == 'B':
						if cur_restriction != "":
							item["restriction"].append(cur_restriction[0:-1])
							cur_restriction = ""
						cur_restriction += restriction_line_words[0] + " "
					elif restriction_line_words[5] == "I":
						cur_restriction += restriction_line_words[0] + " "
					elif restriction_line_words[5] == "E":
						cur_restriction += restriction_line_words[0] + " "
						item["restriction"].append(cur_restriction[0:-1])
						cur_restriction = ""
					elif restriction_line_words[5] == "O":
						if cur_restriction != "":
							item["restriction"].append(cur_restriction[0:-1])
							cur_restriction = ""
				elif agent_line == '\n':
					item[":"] = item[":"][0:-1]
					if len(item["agent"]) > 1:
						item["agent"] = item["agent"][0:-1]
					if len(item["operation"]) > 1:
						item["operation"] = item["operation"][0:-1]
					event.append(item)
					rl[i]["event"] = event
					# print("i in the last sentence : " + str(i))
					j += 1



					break
				j += 1
	with open(outPath, 'w') as f:
		json.dump(rl, f)

def format_sent(sent):
	sent = sent.replace("( ", "(")
	sent = sent.replace(" )", ")")
	sent = sent.replace(" / ", "/")
	sent = sent.replace(" 's", "'s")
	sent = sent.replace(" - ", "-")
	sent = sent.replace(" -", "-")
	sent = sent.replace("- ", "-")
	sent = sent.replace("/ ", "/")
	sent = sent.replace(" ,", ",")
	sent = sent.lower()
	return sent



def is_substr(str1, str2):
	list1, list2, j = str1.split(), str2.split(), 0
	for i in list1:
		while i != list2[j]:
			j += 1
			if j >= len(list2):
				return False
	return True

def is_sublist(list1, list2):
	flag = 0
	for one in list1:
		for another in list2:
			if one in another:
				flag = 1
				break
	for one in list2:
		for another in list1:
			if one in another:
				flag = 1
				break

	if flag == 1:
		return True
	else:
		return False


def test_restriction(outPath):
	restriction_list = json.load(open(outPath))
	condition_list = [',', '.', "when", "if"]
	none_list = ["when", "if", "to", "of", "'s"]
	special_words = ['-', '/']
	# 处理主句中的restriction
	for i, one in enumerate(restriction_list):
		one["restriction"] = []
		if one["restriction"] == [] or one["restriction"] != []:
			cur_restriction = ""
			res_flag = 0
			doc = nlp(one[":"])
			for sent in doc.sentences:
				for j, word in enumerate(sent.words):
					if res_flag == 1:
						if word.lemma in condition_list or word.deprel == "aux" or word.deprel == "root":
							res_flag = 0
							cur_restriction = format_sent(cur_restriction)
							one["restriction"].append(cur_restriction[0:-1])
							cur_restriction = ""
						else:
							if word.lemma in special_words:
								cur_restriction = cur_restriction[0:-1] + word.text
							else:
								cur_restriction += word.text + " "
					elif word.deprel == "advmod" and word.xpos == "RB" and word.text != "not":
						one["restriction"].append(word.text)
						cur_restriction = ""
					elif word.deprel == "case" or word.deprel == "mark" or word.text == "maximum" or (word.xpos == "DT" and word.deprel == "advmod"):
						if word.lemma not in none_list:
							cur_restriction += word.text + " "
							res_flag = 1
	# 处理触发事件中的restriction
	for i, one in enumerate(restriction_list):
		if "event" in one and one["event"][0]["restriction"] == []:
			cur_restriction = ""
			res_flag = 0
			doc = nlp(one["event"][0][":"])
			for sent in doc.sentences:
				for j, word in enumerate(sent.words):
					if res_flag == 1:
						if word.lemma in condition_list or word.deprel == "aux" or word.deprel == "root":
							res_flag = 0
							cur_restriction = format_sent(cur_restriction)
							one["event"][0]["restriction"].append(cur_restriction[0:-1])
							cur_restriction = ""
						elif j == len(sent.words)-1:
							cur_restriction += word.text
							res_flag = 0
							one["event"][0]["restriction"].append(cur_restriction)
							cur_restriction = ""
						else:
							if word.lemma in special_words:
								cur_restriction = cur_restriction[0:-1] + word.text
							else:
								cur_restriction += word.text + " "
					elif word.deprel == "advmod" and word.xpos == "RB" and word.text != "not":
						one["event"][0]["restriction"].append(word.text)
						cur_restriction = ""
					elif word.deprel == "case" or (word.xpos == "DT" and word.deprel == "advmod"):
						if word.lemma not in none_list:
							cur_restriction += word.text + " "
							res_flag = 1

	json.dump(restriction_list, open(outPath, 'w'))

def check(prePath, outPath):
	agent_error = 0
	operation_error = 0
	input_error = 0
	output_error = 0
	restriction_error = 0
	event_error = 0

	restriction_num = 0
	input_num = 0
	output_num = 0
	agent_num = 0
	operation_num = 0
	event_num = 0


	pass_input = 0
	correct_num = 0
	f_pre = json.load(open(prePath))
	f_out = json.load(open(outPath))
	for i, item in enumerate(f_pre):
		one = {
			"#": str(i + 1),
			":": item[":"],
			"agent": "",
			"operation": "",
			"input": [],
			"output": [],
			"restriction": []
		}
		flag = 0
		print('------------------------------')
		if "agent" in item and "entity" in item["agent"]:

			if len(f_out[i]["agent"]) == 0:
				print("agent wrong not rectified")
			agent_num += 1
			one["agent"] = item["agent"]["entity"]
			print(one["agent"])
			print(f_out[i]["agent"])
			if (one["agent"].lower() not in f_out[i]["agent"].lower() and f_out[i]["agent"].lower() not in one[
				"agent"].lower()) or len(f_out[i]["agent"]) == 0:
				flag += 1
				agent_error += 1
				print("agent wrong")

		if "operation" in item:
			operation_num += 1
			one["operation"] = item["operation"]["operation"]
			print(prePath)
			print(one["operation"])
			print(f_out[i]["operation"])
			print(one[":"])
			if f_out[i]["operation"] == "":
				operation_error += 1
				flag += 1
				print("operation wrong")
			elif one["operation"] != "" and f_out[i]["operation"] != "" and not is_substr(f_out[i]["operation"], one["operation"]) and not is_substr(one["operation"], f_out[i]["operation"]):
				test1_sent = nlp(one["operation"]).sentences[0].words
				test2_sent = nlp(f_out[i]["operation"]).sentences[0].words

				if len(test1_sent) > len(test2_sent):
					for p in range(len(test2_sent)):
						test1 = test1_sent[p].lemma
						test2 = test2_sent[p].lemma
						if test1 != test2 and test1 not in test2 and not test2 in f_out[i]["operation"]:
							operation_error += 1
							flag += 1
							print("operation wrong")
							break
				else:

					for p in range(len(test1_sent)):
						test1 = test1_sent[p].lemma
						test2 = test2_sent[p].lemma
						if test1 != test2 and test1 not in test2 and not test1 in one["operation"]:
							operation_error += 1
							flag += 1
							print("operation wrong")
							break

		if "input" in item:
			input_num += 1
			one["input"] = [pp["entity"].lower() for pp in item["input"]["()"]]
			f_out[i]["input"] = [format_sent(pp) for pp in f_out[i]["input"]]
			print("input : ")
			print(one["input"])
			print(f_out[i]["input"])
			if sorted(one["input"]) != sorted(f_out[i]["input"]) and not is_sublist(one["input"], f_out[i]["input"]) and not is_sublist(f_out[i]["input"], one["input"]):
				input_error += 1
				flag += 1
				if "be" in one[':']:
					pass_input += 1

				print(one[":"])
				print("input wrong")

		if "output" in f_out[i]:
			print(f_out[i]["output"])
		if "output" in item:
			output_num += 1
			one["output"] = [pp["entity"].lower() for pp in item["output"]["()"]]
			f_out[i]["output"] = [format_sent(pp) for pp in f_out[i]["output"]]
			print("output: ")
			print(one["output"])
			print(f_out[i]["output"])
			if sorted(one["output"]) != sorted(f_out[i]["output"]) and not is_sublist(one["output"], f_out[i]["output"]) and not is_sublist(f_out[i]["output"], one["output"]):
				output_error += 1
				flag += 1
				print("output wrong")


		if "restriction" in item:

			restriction_num += 1
			one["restriction"] = [res_item.lower() for res_item in item["restriction"]["()"]]
			print("restriction: ")
			print(one["restriction"])
			print(f_out[i]["restriction"])
			if sorted(one["restriction"]) != sorted(f_out[i]["restriction"]) and not is_sublist(one["restriction"], f_out[i]["restriction"]) and not is_sublist(f_out[i]["restriction"], one["restriction"]):
				restriction_error += 1
				flag += 1
				print("restriction wrong")

		if "event" in item:
			event_num += 1
			event_flag = 0
			one["event"] = []
			result_event = {
				"agent": "",
				"operation": "",
				"input": [],
				"output": [],
				"restriction": []
			}
			for one_event in item["event"]["()"]:
				if "agent" in one_event:
					if one_event["agent"]["entity"] not in result_event["agent"]:
						result_event["agent"] += one_event["agent"]["entity"] + " "
				if "operation" in one_event:
					result_event["operation"] += one_event["operation"]["operation"] + " "
				if "input" in one_event:
					result_event["input"] += [pp["entity"] for pp in one_event["input"]["()"]]
				if "output" in one_event:
					result_event["output"] += [pp["entity"] for pp in one_event["output"]["()"]]
				if "restriction" in one_event:
					print(one_event["restriction"])
					result_event["restriction"] += one_event["restriction"]["()"]
			if result_event["agent"] != "":
				result_event["agent"] = result_event["agent"][0:-1]
			if result_event["operation"]:
				result_event["operation"] = result_event["operation"][0:-1]
			one["event"].append(result_event)

			if one['event'] != [] and "event" in f_out[i]:
				event_item = one["event"][0]
				f_item = f_out[i]["event"][0]
				# agent

				print("event agent:")
				print(event_item["agent"])
				print(f_item["agent"])
				if event_item["agent"].lower() not in f_item["agent"].lower() and f_item["agent"].lower() not in event_item["agent"].lower():
					event_flag += 1
					print("event agent wrong")
				# operation

				print("event operation:")
				print(event_item["operation"])
				print(f_item["operation"])
				if (event_item["operation"] == "" and f_item["operation"] != "") or (
						f_item["operation"] == "" and event_item["operation"] != ""):
					event_flag += 1
					print("here goes wrong")
					print("event operation wrong")
				elif event_item["operation"] != "" and f_item["operation"] != "" and not is_substr(f_item["operation"], event_item["operation"]) and not is_substr(event_item["operation"], f_item["operation"]):
					test1_sent = nlp(event_item["operation"]).sentences[0].words
					test2_sent = nlp(f_item["operation"]).sentences[0].words
					if len(test1_sent) > len(test2_sent):
						for p in range(len(test2_sent)):
							test1 = test1_sent[p].lemma
							test2 = test2_sent[p].lemma
							if test1 != test2 and test2 not in event_item["operation"] and test2[0:-1] not in event_item["operation"]:
								event_flag += 1
								print("event operation wrong")
								break
					else:
						for p in range(len(test1_sent)):
							test1 = test1_sent[p].lemma
							test2 = test2_sent[p].lemma

							if test1 != test2 and test1 not in f_item["operation"] and test1[0:-1] not in f_item["operation"]:
								event_flag += 1
								print("event operation wrong")
								break

				# input
				print("event input:")
				print(event_item["input"])
				print(f_item["input"])
				if sorted(event_item["input"]) != sorted(f_item["input"]) and not is_sublist(event_item["input"],
																							 f_item[
																								 "input"]) and not is_sublist(
						f_item["input"], event_item["input"]):
					event_flag += 1
					print("event input wrong")

				# output
				print("event output:")
				print(event_item["output"])
				print(f_item["output"])
				if sorted(event_item["output"]) != sorted(f_item["output"]) and not is_sublist(event_item["output"],
																							   f_item[
																								   "output"]) and not is_sublist(
						f_item["output"], event_item["output"]):
					event_flag += 1
					print("event output wrong")

				print("event restriction:")
				print(event_item["restriction"])
				print(f_item["restriction"])
				# restriction
				if sorted(event_item["restriction"]) != sorted(f_item["restriction"]) and not is_sublist(
						event_item["restriction"], f_item["restriction"]) and not is_sublist(f_item["restriction"],
																							 event_item["restriction"]):
					event_flag += 1
					print("event restriction wrong")

				if event_flag > 1:
					flag += 1
					event_error += 1
					print("event wrong")
			elif one['event'] != [] and not "event" in f_out[i]:
				flag += 1
				event_error += 1
				print("event wrong")

		if flag == 0:
			correct_num += 1
		
		print(flag)
		print('------------------------------')

		print()

	final_json_list = []
	try:
		with open('/Users/tuanz_lu/PycharmProjects/Pytorch-learning/exp/tuple_eval.json', 'r') as f:
			final_json_list = json.load(f)
			print(final_json_list)
	except json.decoder.JSONDecodeError:
		final_json_list = []
		print("empty!")

	final_json = {
		"data_id": str(len(final_json_list)+1),
		"prePath": prePath,
		"total": str(len(f_pre)),
		"correct_num": str(correct_num),
		"val_accuracy: ": str(correct_num / len(f_pre)),
		"agent": {
			"agent_error": str(agent_error),
			"accuracy":str((agent_num-agent_error)/agent_num),
			"correct_num": str(agent_num-agent_error),
			"total": str(agent_num)
		},
		"operation": {
			"operation_error": str(operation_error),
			"accuracy": str((operation_num - operation_error) / operation_num),
			"correct_num": str(operation_num - operation_error),
			"total": str(operation_num)
		},
		"input": {
			"input_error": str(input_error),
			"accuracy": str((input_num - input_error) / input_num),
			"correct_num": str(input_num - input_error),
			"total": str(input_num)
		},
		"output": {
			"output_error": str(output_error),
			"accuracy": str((output_num - output_error) / output_num),
			"correct_num": str(output_num - output_error),
			"total": str(output_num)
		},
		"restriction": {
			"restriction_error": str(restriction_error),
			"accuracy": str((restriction_num - restriction_error) / restriction_num),
			"correct_num": str(restriction_num - restriction_error),
			"total": str(restriction_num)
		},
		"event": {
			"event_error": str(event_error),
			"accuracy": str((event_num - event_error) / event_num),
			"correct_num": str(event_num - event_error),
			"total": str(event_num)
		}
	}
	final = open('/Users/tuanz_lu/PycharmProjects/Pytorch-learning/exp/tuple_eval.json', 'w')

	final_json_list.append(final_json)
	json.dump(final_json_list, final)

	return {
		"total": len(f_pre),
		"correct_num": correct_num,
		"accuracy": correct_num / len(f_pre)
	}




exp_list = ['1', '2', '3', '4', '5']
temp = ['1']
c_list = ['10']

tag_list = ['E', 'A', 'B', 'C', 'D']

for i, cur_exp in enumerate(exp_list):
	for j, cur_temp in enumerate(temp):
		for p, cur_c in enumerate(c_list):
			cur_tag = ''
			cur_tag += cur_exp + cur_temp+cur_c
			tag = tag_list[i]
			agent_path = file_path + '/output/agent_' + cur_tag + '.output'
			operation_path = file_path + '/output/operation_' + cur_tag + '.output'
			input_path = file_path + '/output/input_' + cur_tag + '.output'
			output_path = file_path + '/output/output_' + cur_tag + '.output'
			restriction_path = file_path + '/output/restriction_' + cur_tag + '.output'
			event_agent_path = file_path + '/output/event_agent_' + cur_tag + '.output'
			event_operation_path = file_path + '/output/event_operation_' + cur_tag + '.output'
			event_input_path = file_path + '/output/event_input_' + cur_tag + '.output'
			event_output_path = file_path + '/output/event_output_' + cur_tag + '.output'
			event_restriction_path = file_path + '/output/event_restriction_' + cur_tag + '.output'

			get_agent(inPath=agent_path, outPath=file_path + '/back/test_'+ tag + '.json')
			get_operation(inPath=operation_path, outPath=file_path + '/back/test_'+ tag + '.json')
			get_input(inPath=input_path, outPath=file_path + '/back/test_'+ tag + '.json')
			get_output(inPath=output_path, outPath=file_path + '/back/test_'+ tag + '.json')
			# get_restriction(inPath=restriction_path, outPath=file_path + '/back/test_'+ tag + '.json')

			get_event(event_agent_path, event_operation_path, event_input_path, event_output_path,
					  event_restriction_path,
					  outPath=file_path + '/back/test_' + tag + '.json')
			test_restriction(outPath=file_path + '/back/test_'+ tag + '.json')



			# final.write( cur_tag+'\n')
			final_item = check(prePath='/Users/tuanz_lu/PycharmProjects/Pytorch-learning/json_data/five/' + tag + '_data.json',
				  outPath=file_path + '/back/test_'+ tag + '.json')
			final_list.append(final_item)

