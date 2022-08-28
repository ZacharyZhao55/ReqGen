#!/bin/sh
crf_path=/root/Desktop/uav_req_gen_unilm_v2/unilm/src/crf_project/CRF++-0.58
pre_data_path=/root/Desktop/uav_req_gen_unilm_v2/unilm/src/crf_project/exp/pre/


"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/agent_model_5110 "$pre_data_path"agent_pre_gannt.txt > "$crf_path"/example/exp/gannt/agent_gannt_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/operation_model_5110 "$pre_data_path"operation_pre_gannt.txt > "$crf_path"/example/exp/gannt/operation_gannt_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/restriction_model_5110 "$pre_data_path"restriction_pre_gannt.txt > "$crf_path"/example/exp/gannt/restriction_gannt_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/output_model_5110 "$pre_data_path"output_pre_gannt.txt > "$crf_path"/example/exp/gannt/output_gannt_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/input_model_5110 "$pre_data_path"input_pre_gannt.txt > "$crf_path"/example/exp/gannt/input_gannt_1110.output

"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/event_agent_model_5110 "$pre_data_path"event_pre_agent_gannt.txt > "$crf_path"/example/exp/gannt/event_agent_gannt_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/event_operation_model_5110 "$pre_data_path"event_pre_operation_gannt.txt > "$crf_path"/example/exp/gannt/event_operation_gannt_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/event_input_model_5110 "$pre_data_path"event_pre_input_gannt.txt > "$crf_path"/example/exp/gannt/event_input_gannt_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/event_output_model_5110 "$pre_data_path"event_pre_output_gannt.txt > "$crf_path"/example/exp/gannt/event_output_gannt_1110.output
"$crf_path"/crf_test  -m "$crf_path"/example/exp/model/event_restriction_model_5110 "$pre_data_path"event_pre_restriction_gannt.txt > "$crf_path"/example/exp/gannt/event_restriction_gannt_1110.output