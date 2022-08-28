import queue

from graphviz import Digraph
import json


# 得到交互关系总图
def get_input_total_pic(file, input_path):
	g = Digraph(name= file+'/交互关系总图',format='png')

	file_data = json.load(open(input_path))
	for i, item in enumerate(file_data):
		g.node(name=item["node_id"],color='black')
		for j, edge in enumerate(item["edges"]):
			g.node(name=edge["output"], color='black')
			g.edge(edge["output"], edge["input"], color='green')

	g.view()


# 得到自定义子图
def get_sep_pic(file, input_path):
	node_list = []
	if file == "gannt":
		# 观察总关系图拆分出参与展示的节点值
		node_list = [ [3,69,70,72,5,9,52,78,4,7,6,86,14,17],
			 [19,20,21,22,23,24,25,26,27,35,46],
			 [30,31,38,33,42,79,80,47,48,65,68,71,75]]
	for i, one_list in enumerate(node_list):
		g = Digraph(name=file+'/交互关系分块图00'+str(i),format='png')

		file_data = json.load(open(input_path))
		sep_data = [item for item in file_data if int(item["node_id"]) in one_list]
		for i, item in enumerate(sep_data):
			g.node(name=item["node_id"], color='black')
			for j, edge in enumerate(item["edges"]):
				g.edge(edge["output"], edge["input"], color='green')

		g.view()


# 得到前提依赖关系总图
def get_event_total_pic(file, event_path):
	g = Digraph(name= file+'/前提依赖关系总图', format='png')
	file_data = json.load(open(event_path))
	for i, item in enumerate(file_data):
		g.node(name=item["req1_id"], color='black')
		g.node(name=item["req2_id"], color='black')
		g.edge(item["req1_id"], item["req2_id"], color='blue')

	g.view()


# 得到总关系图
def get_total(file, input_path, event_path):
	g = Digraph(name=file+'/关系总图', format='png')
	input_data = json.load(open(input_path))
	event_data = json.load(open(event_path))
	for i, item in enumerate(event_data):
		g.node(name=item["req1_id"], color='black')
		g.node(name=item["req2_id"], color='black')
		g.edge(item["req2_id"], item["req1_id"], color='blue')
	for i, item in enumerate(input_data):
		g.node(name=item["node_id"],color='black')
		for j, edge in enumerate(item["edges"]):
			g.node(name=edge["output"], color='black')
			g.edge(edge["output"], edge["input"], color='green')

	g.view()


# 得到总关系网，用于后续计算需求优先级
def get_all_relation(input_path, event_path, output_path):
	relation_list = []
	input_data = json.load(open(input_path))
	event_data = json.load(open(event_path))
	for i, item in enumerate(input_data):
		for one_edge in item["edges"]:
			relation_item = {
				"req1_id": one_edge["output"],
				"req1": one_edge["output_req"],
				"req2_id": one_edge["input"],
				"req2": one_edge["input_req"]
			}
			relation_list.append(relation_item)

	for i, item in enumerate(event_data):
		relation_item = {
			"req1_id": item["req2_id"],
			"req1": item["req2"],
			"req2_id": item["req1_id"],
			"req2": item["req1"]
		}
		relation_list.append(relation_item)
	with open(output_path, 'w') as f:
		json.dump(relation_list, f)


# 得到需求优先级可视化结果
def draw_grade(file, input_path, id):
	g = Digraph('structs', filename= file+'/'+id+'_需求优先级图', format='png', node_attr={'shape': 'record'})
	file_data = json.load(open(input_path))
	for i, one_list in enumerate(file_data):
		print(one_list)
		one_list = [str(j) for j in one_list]
		g.node('struct'+str(i), '|'.join(one_list))

	for i in range(len(file_data)-1):
		g.edge('struct'+str(i), 'struct'+str(i+1))
	g.view()


# 获取总的需求优先顺序和子图中的需求优先顺序
def get_req_grade(file, len, input_path, output_easy_path, output_path):
	grade_list = []
	rel_data = json.load(open(input_path))
	out_cnt = [0 for i in range(len)]
	in_cnt = [0 for i in range(len)]
	# count 出入度
	for i, item in enumerate(rel_data):
		out_cnt[int(item["req1_id"])-1] += 1
		in_cnt[int(item["req2_id"])-1] += 1

	nodes = []
	for i in range(len):
		node_item = {
			"id": i+1,
			"in": in_cnt[i],
			"out": out_cnt[i]
		}
		nodes.append(node_item)
	cnt_num = 0

	# 按照入度为0拆分
	while cnt_num < len:
		node_list = []
		for i in range(len):
			if in_cnt[i] == 0:
				cnt_num += 1
				in_cnt[i] -= 1
				node_list.append(nodes[i])
				print(i)
				for item in rel_data:
					if int(item["req1_id"])-1 == i:
						in_cnt[int(item["req2_id"])-1] -= 1
		grade_list.append(node_list)

	# print(grade_list)
	easy_list = []

	for i, one_list in enumerate(grade_list):
		grade_list[i] = sorted(one_list, key=lambda k: k['out'], reverse=True)
		o_list = [item["id"] for item in grade_list[i]]
		easy_list.append(o_list)
	# print(grade_list)

	with open(output_easy_path, 'w') as fp:
		json.dump(easy_list, fp)

	with open(output_path, 'w') as fo:
		json.dump(grade_list, fo)


	pic_list = json.load(open(root_path+'/priority'+file+'seperate_pic.json'))

	one_id_list = []
	no = 1
	# get seperate pics grade
	for id, one in enumerate(pic_list):
		if len(one) == 1:
			one_id_list.append(one[0])
			continue
		grade_list = []
		out_cnt = [0 for i in range(len)]
		in_cnt = [0 for i in range(len)]
		# count 出入度
		for i, item in enumerate(rel_data):
			out_cnt[int(item["req1_id"]) - 1] += 1
			in_cnt[int(item["req2_id"]) - 1] += 1

		nodes = []
		for i in range(len):
			node_item = {
				"id": i + 1,
				"in": in_cnt[i],
				"out": out_cnt[i]
			}
			nodes.append(node_item)
		cnt_num = 0

		# 按照入度为0拆分
		while cnt_num < len(one):
			node_list = []
			for i in range(len):
				if in_cnt[i] == 0 and str(i+1) in one:
					cnt_num += 1
					in_cnt[i] -= 1
					node_list.append(nodes[i])
					print(i)
					for item in rel_data:
						if int(item["req1_id"]) - 1 == i:
							in_cnt[int(item["req2_id"]) - 1] -= 1
			grade_list.append(node_list)

		# print(grade_list)
		easy_list = []

		for i, one_list in enumerate(grade_list):
			grade_list[i] = sorted(one_list, key=lambda k: k['out'], reverse=True)
			o_list = [item["id"] for item in grade_list[i]]
			easy_list.append(o_list)

		with open(root_path+'/priority'+file+''+str(no)+'_pic_easy_grade.json', 'w') as f:
			json.dump(easy_list, f)

		with open(root_path+'/priority'+file+''+str(no)+'_pic_grade.json', 'w') as fo:
			json.dump(grade_list, fo)

		no += 1

	out_cnt = [0 for i in range(len)]
	in_cnt = [0 for i in range(len)]
	# count 出入度
	for i, item in enumerate(rel_data):
		out_cnt[int(item["req1_id"]) - 1] += 1
		in_cnt[int(item["req2_id"]) - 1] += 1

	nodes = []
	for i in range(len):
		node_item = {
			"id": i + 1,
			"in": in_cnt[i],
			"out": out_cnt[i]
		}
		nodes.append(node_item)

	one_list = [nodes[int(i)-1] for i in one_id_list]
	one_list = sorted(one_list, key=lambda k: k['out'], reverse=True)
	one_id_list = [[item["id"] for item in one_list]]
	one_list = [one_list]

	with open(root_path+'/priority'+file+'single_easy_grade.json', 'w') as f:
		json.dump(one_id_list, f)

	with open(root_path+'/priority'+file+'single_grade.json', 'w') as f:
		json.dump(one_list, f)


# 获取跟当前id节点有边的节点值
def get_relate(data, id):
	list = []
	print("get_relate id : "+ str(id))
	for item in data:
		if item["req1_id"] == str(id):
			list.append(item["req2_id"])
		if item["req2_id"] == str(id):
			list.append(item["req1_id"])
	return list


# 拆分子图
def get_seperate(len, input_path, output_sep_path):
	rel_data = json.load(open(input_path))
	print(" 拆分子图 ")
	data = []
	for i, item in enumerate(rel_data):
		addItem = {
			"req1_id": item["req2_id"],
			"req2_id": item["req1_id"]
		}
		data.append(item)
		data.append(addItem)

	# 拆分子图
	pic_list = []
	flag = [0 for i in range(len)]
	for i in range(len):
		node_list = []
		if flag[i] == 0:
			q = queue.Queue()
			q.put(str(i + 1))
			print(i + 1)
			node_list.append(str(i + 1))
			while not q.empty():
				top = q.get()
				if flag[int(top) - 1] == 0:
					flag[int(top) - 1] = 1
					tmp = get_relate(data, top)
					node_list += tmp
					for item in tmp:
						if flag[int(item) - 1] == 0:
							q.put(item)

			node_list = list(set(node_list))
			print(node_list)
			print()
			pic_list.append(node_list)

	with open(output_sep_path, 'w') as fout:
		json.dump(pic_list, fout)




root_path="/Users/tuanz_lu/PycharmProjects/Pytorch-learning"

# 进行可视化的文件目录
file_list = ['gannt']
for file in file_list:
	req_list = json.load(open(root_path + '/gannt/back/test_' + file + '.json'))
	len = len(req_list)

	# get_sep_pic(file, root_path+'/priority/'+file+'/wmd_input_to_output.json')
	# get_input_total_pic(file, root_path+'/priority/'+file+'/wmd_input_to_output.json')
	# get_event_total_pic(file, root_path+'/priority/'+file+'/wmd_event.json')
	# get_total(file, root_path+'/priority/'+file+'/wmd_input_to_output.json', root_path+'/priority/'+file+'/wmd_event.json')


	# get_all_relation(root_path+'/priority/'+file+'/wmd_input_to_output.json', root_path+'/priority/'+file+'/wmd_event.json', root_path+'/priority/'+file+'/all_relation.json')

	# get_seperate(len, root_path+'/priority'+file+'all_relation.json', root_path+'/priority'+file+'seperate_pic.json')

	# get_req_grade(file, len,root_path+'/priority'+file+'all_relation.json', root_path+'/priority'+file+'req_easy_grade.json', root_path+'/priority'+file+'req_grade.json')


	# 根据拓扑排序结合出度降序排序拆分的子图，按子图绘制优先顺序图
	# draw_grade('gannt', root_path+'/priority/'+file+'/1_pic_easy_grade.json', '1')
	# draw_grade('gannt', root_path+'/priority/' + file + '/2_pic_easy_grade.json', '2')
	# draw_grade('gannt', root_path+'/priority/' + file + '/3_pic_easy_grade.json', '3')
	# draw_grade('gannt', root_path+'/priority/' + file + '/4_pic_easy_grade.json', '4')
	# draw_grade('gannt', root_path+'/priority/' + file + '/5_pic_easy_grade.json', '5')
	# draw_grade('gannt', root_path+'/priority/' + file + '/6_pic_easy_grade.json', '6')
	# draw_grade('gannt', root_path+'/priority/' + file + '/7_pic_easy_grade.json', '7')
	# draw_grade('gannt', root_path+'/priority/' + file + '/single_easy_grade.json', 'single')



