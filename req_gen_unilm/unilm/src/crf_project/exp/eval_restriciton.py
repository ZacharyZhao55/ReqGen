import datetime
import json


with open('/Users/tuanz_lu/PycharmProjects/Pytorch-learning/exp/tuple_eval.json') as f:
	result_list = json.load(f)
	final_json = {
		"data_id": "avg_result",
		"loop_times": len(result_list),
		"total": 0,
		"val_accuracy": 0,
		"correct_num": 0,
		"agent": {
			"agent_error": 0,
			"accuracy": 0,
			"correct_num": 0,
			"total":0
		},
		"operation": {
			"operation_error": 0,
			"accuracy": 0,
			"correct_num": 0,
			"total": 0
		},
		"input": {
			"input_error":  0,
			"accuracy":  0,
			"correct_num": 0,
			"total":0
		},
		"output": {
			"output_error":  0,
			"accuracy": 0,
			"correct_num": 0,
			"total":  0
		},
		"restriction": {
			"restriction_error":  0,
			"accuracy":  0,
			"correct_num":  0,
			"total":  0
		},
		"event": {
			"event_error":  0,
			"accuracy": 0,
			"correct_num": 0,
			"total":  0
		}
	}


	for i, item in enumerate(result_list):
		if i == 1:
			final_json["total"] = int(item["total"])


		final_json["correct_num"] = final_json["correct_num"] + int(item["correct_num"])
		# print(final_json['val_accuracy'])
		# print(item['val_accuracy'])
		# final_json["val_accuracy"] = final_json["val_accuracy"] + float(item["val_accuracy"])

		final_json["agent"]["correct_num"] = final_json["agent"]["correct_num"] + int(item["agent"]["correct_num"])
		final_json["agent"]["agent_error"] = final_json["agent"]["agent_error"] + int(item["agent"]["agent_error"])
		final_json["agent"]["total"] = final_json["agent"]["total"] + int(item["agent"]["total"])
		final_json["agent"]["accuracy"] = final_json["agent"]["accuracy"] + float(item["agent"]["accuracy"])

		final_json["operation"]["correct_num"] = final_json["operation"]["correct_num"] + int(
			item["operation"]["correct_num"])
		final_json["operation"]["operation_error"] = final_json["operation"]["operation_error"] + int(
			item["operation"]["operation_error"])
		final_json["operation"]["total"] = final_json["operation"]["total"] + int(item["operation"]["total"])
		final_json["operation"]["accuracy"] = final_json["operation"]["accuracy"] + float(item["operation"]["accuracy"])

		final_json["input"]["correct_num"] = final_json["input"]["correct_num"] + int(item["input"]["correct_num"])
		final_json["input"]["input_error"] = final_json["input"]["input_error"] + int(item["input"]["input_error"])
		final_json["input"]["total"] = final_json["input"]["total"] + int(item["input"]["total"])
		final_json["input"]["accuracy"] = final_json["input"]["accuracy"] + float(item["input"]["accuracy"])

		final_json["output"]["correct_num"] = final_json["output"]["correct_num"] + int(item["output"]["correct_num"])
		final_json["output"]["output_error"] = final_json["output"]["output_error"] + int(
			item["output"]["output_error"])
		final_json["output"]["total"] = final_json["output"]["total"] + int(item["output"]["total"])
		final_json["output"]["accuracy"] = final_json["output"]["accuracy"] + float(item["output"]["accuracy"])

		final_json["restriction"]["correct_num"] = final_json["restriction"]["correct_num"] + int(
			item["restriction"]["correct_num"])
		final_json["restriction"]["restriction_error"] = final_json["restriction"]["restriction_error"] + int(
			item["restriction"]["restriction_error"])
		final_json["restriction"]["total"] = final_json["restriction"]["total"] + int(item["restriction"]["total"])
		final_json["restriction"]["accuracy"] = final_json["restriction"]["accuracy"] + float(
			item["restriction"]["accuracy"])

		final_json["event"]["correct_num"] = final_json["event"]["correct_num"] + int(item["event"]["correct_num"])
		final_json["event"]["event_error"] = final_json["event"]["event_error"] + int(item["event"]["event_error"])
		final_json["event"]["total"] = final_json["event"]["total"] + int(item["event"]["total"])
		final_json["event"]["accuracy"] = final_json["event"]["accuracy"] + float(item["event"]["accuracy"])


	final_json["correct_num"] = round(final_json["correct_num"] / len(result_list))
	final_json["val_accuracy"] = final_json["correct_num"] / final_json["total"]

	final_json["agent"]["agent_error"] = round(final_json["agent"]["agent_error"] / len(result_list))
	final_json["agent"]["accuracy"] = final_json["agent"]["correct_num"] / final_json["agent"]["total"]
	final_json["agent"]["correct_num"] = round(final_json["agent"]["correct_num"] / len(result_list))
	final_json["agent"]["total"] = round(final_json["agent"]["total"] / len(result_list))

	final_json["operation"]["operation_error"] = round(final_json["operation"]["operation_error"] / len(result_list))
	final_json["operation"]["accuracy"] = final_json["operation"]["correct_num"] / final_json["operation"]["total"]
	final_json["operation"]["correct_num"] = round(final_json["operation"]["correct_num"] / len(result_list))
	final_json["operation"]["total"] = round(final_json["operation"]["total"] / len(result_list))

	final_json["input"]["input_error"] = round(final_json["input"]["input_error"] / len(result_list))
	final_json["input"]["accuracy"] = final_json["input"]["correct_num"] / final_json["input"]["total"]
	final_json["input"]["correct_num"] = round(final_json["input"]["correct_num"] / len(result_list))
	final_json["input"]["total"] = round(final_json["input"]["total"] / len(result_list))

	final_json["output"]["output_error"] = round(final_json["output"]["output_error"] / len(result_list))
	final_json["output"]["accuracy"] = final_json["output"]["correct_num"] / final_json["output"]["total"]
	final_json["output"]["correct_num"] = round(final_json["output"]["correct_num"] / len(result_list))
	final_json["output"]["total"] = round(final_json["output"]["total"] / len(result_list))

	final_json["restriction"]["restriction_error"] = round(
		final_json["restriction"]["restriction_error"] / len(result_list))
	final_json["restriction"]["accuracy"] = final_json["restriction"]["correct_num"] / final_json["restriction"][
		"total"]
	final_json["restriction"]["correct_num"] = round(final_json["restriction"]["correct_num"] / len(result_list))
	final_json["restriction"]["total"] = round(final_json["restriction"]["total"] / len(result_list))

	final_json["event"]["event_error"] = round(final_json["event"]["event_error"] / len(result_list))
	final_json["event"]["accuracy"] = final_json["event"]["correct_num"] / final_json["event"]["total"]
	final_json["event"]["correct_num"] = round(final_json["event"]["correct_num"] / len(result_list))
	final_json["event"]["total"] = round(final_json["event"]["total"] / len(result_list))

	print(final_json)
	now = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
	with open('/Users/tuanz_lu/PycharmProjects/Pytorch-learning/exp/avg_result'+ now+ '.json', 'w') as fout:
		json.dump(final_json, fout)