#!/bin/sh
crf_path=/root/Desktop/crf_project/crf_project/CRF++-0.58

"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/agent_model_5110 "$crf_path"/example/exp/pre_data/agent_pre_worldvista.txt > "$crf_path"/example/exp/gannt/agent_worldvista_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/operation_model_5110 "$crf_path"/example/exp/pre_data/operation_pre_worldvista.txt > "$crf_path"/example/exp/gannt/operation_worldvista_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/restriction_model_5110 "$crf_path"/example/exp/pre_data/restriction_pre_worldvista.txt > "$crf_path"/example/exp/gannt/restriction_worldvista_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/output_model_5110 "$crf_path"/example/exp/pre_data/output_pre_worldvista.txt > "$crf_path"/example/exp/gannt/output_worldvista_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/input_model_5110 "$crf_path"/example/exp/pre_data/input_pre_worldvista.txt > "$crf_path"/example/exp/gannt/input_worldvista_1110.output

"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/event_agent_model_5110 "$crf_path"/example/exp/pre_data/event_pre_agent_worldvista.txt > "$crf_path"/example/exp/gannt/event_agent_worldvista_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/event_operation_model_5110 "$crf_path"/example/exp/pre_data/event_pre_operation_worldvista.txt > "$crf_path"/example/exp/gannt/event_operation_worldvista_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/event_input_model_5110 "$crf_path"/example/exp/pre_data/event_pre_input_worldvista.txt > "$crf_path"/example/exp/gannt/event_input_worldvista_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/event_output_model_5110 "$crf_path"/example/exp/pre_data/event_pre_output_worldvista.txt > "$crf_path"/example/exp/gannt/event_output_worldvista_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/event_restriction_model_5110 "$crf_path"/example/exp/pre_data/event_pre_restriction_worldvista.txt > "$crf_path"/example/exp/gannt/event_restriction_worldvista_1110.output