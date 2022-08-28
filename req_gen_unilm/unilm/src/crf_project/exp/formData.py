str_list = ['agent_pre', 'operation_pre', 'input_pre', 'output_pre', 'restriction_pre', 'event_pre_agent',
			'event_pre_operation', 'event_pre_input', 'event_pre_output', 'event_pre_restriction']
out_list = ['agent', 'operation', 'input', 'output', 'restriction', 'event_agent', 'event_operation', 'event_input',
			'event_output', 'event_restriction']

for i, ss in enumerate(str_list):
	yy = out_list[i]
	f1 = open('/Users/tuanz_lu/PycharmProjects/Pytorch-learning/exp/pre/' + ss + '_A.txt')
	f2 = open('/Users/tuanz_lu/PycharmProjects/Pytorch-learning/exp/pre/' + ss + '_B.txt')
	f3 = open('/Users/tuanz_lu/PycharmProjects/Pytorch-learning/exp/pre/' + ss + '_C.txt')
	f4 = open('/Users/tuanz_lu/PycharmProjects/Pytorch-learning/exp/pre/' + ss + '_D.txt')
	f5 = open('/Users/tuanz_lu/PycharmProjects/Pytorch-learning/exp/pre/' + ss + '_E.txt')

	r1 = f1.readlines()
	r2 = f2.readlines()
	r3 = f3.readlines()
	r4 = f4.readlines()
	r5 = f5.readlines()
	r = r1 + r2 + r3 + r4
	with open('/Users/tuanz_lu/PycharmProjects/Pytorch-learning/exp/dataset/first_' + yy + '.txt', 'w') as f:
		for line in r:
			f.write(line)

	r = r2 + r3 + r4 + r5
	with open('/Users/tuanz_lu/PycharmProjects/Pytorch-learning/exp/dataset/second_' + yy + '.txt', 'w') as f:
		for line in r:
			f.write(line)

	r = r1 + r3 + r4 + r5
	with open('/Users/tuanz_lu/PycharmProjects/Pytorch-learning/exp/dataset/third_' + yy + '.txt', 'w') as f:
		for line in r:
			f.write(line)

	r = r1 + r2 + r4 + r5
	with open('/Users/tuanz_lu/PycharmProjects/Pytorch-learning/exp/dataset/four_' + yy + '.txt', 'w') as f:
		for line in r:
			f.write(line)

	r = r1 + r2 + r3 + r5
	with open('/Users/tuanz_lu/PycharmProjects/Pytorch-learning/exp/dataset/five_' + yy + '.txt', 'w') as f:
		for line in r:
			f.write(line)
