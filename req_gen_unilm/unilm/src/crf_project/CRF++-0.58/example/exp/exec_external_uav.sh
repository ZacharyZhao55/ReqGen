#!/bin/sh
crf_path=/root/Desktop/crf_project/crf_project/CRF++-0.58

"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/agent_model_4110 "$crf_path"/example/exp/pre_data/agent_pre_uav.txt > "$crf_path"/example/exp/gannt/agent_uav_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/operation_model_4110 "$crf_path"/example/exp/pre_data/operation_pre_uav.txt > "$crf_path"/example/exp/gannt/operation_uav_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/restriction_model_4110 "$crf_path"/example/exp/pre_data/restriction_pre_uav.txt > "$crf_path"/example/exp/gannt/restriction_uav_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/output_model_4110 "$crf_path"/example/exp/pre_data/output_pre_uav.txt > "$crf_path"/example/exp/gannt/output_uav_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/input_model_4110 "$crf_path"/example/exp/pre_data/input_pre_uav.txt > "$crf_path"/example/exp/gannt/input_uav_1110.output

"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/event_agent_model_4110 "$crf_path"/example/exp/pre_data/event_pre_agent_uav.txt > "$crf_path"/example/exp/gannt/event_agent_uav_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/event_operation_model_4110 "$crf_path"/example/exp/pre_data/event_pre_operation_uav.txt > "$crf_path"/example/exp/gannt/event_operation_uav_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/event_input_model_4110 "$crf_path"/example/exp/pre_data/event_pre_input_uav.txt > "$crf_path"/example/exp/gannt/event_input_uav_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/event_output_model_4110 "$crf_path"/example/exp/pre_data/event_pre_output_uav.txt > "$crf_path"/example/exp/gannt/event_output_uav_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/event_restriction_model_4110 "$crf_path"/example/exp/pre_data/event_pre_restriction_uav.txt > "$crf_path"/example/exp/gannt/event_restriction_uav_1110.output