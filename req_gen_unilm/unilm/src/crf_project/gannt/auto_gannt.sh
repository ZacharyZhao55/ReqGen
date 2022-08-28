#!/bin/bash

structure_path=/root/Desktop/uav_req_gen_unilm_v2/unilm/src/crf_project/gannt
crf_path=/root/Desktop/uav_req_gen_unilm_v2/unilm/src/crf_project/CRF++-0.58
pre_path=/root/Desktop/uav_req_gen_unilm_v2/unilm/src/crf_project/exp/pre/

# 执行预处理程序
#python "$structure_path"/no_exp/operation_pre_stanza.py
#python "$structure_path"/no_exp/agent_pre.py
#python "$structure_path"/no_exp/input_pre.py
#python "$structure_path"/no_exp/output_pre.py
#python "$structure_path"/no_exp/restriction_pre.py
#python "$structure_path"/no_exp/event_pre_stanza.py
#python "$structure_path"/no_exp/str_pre.py

# 将处理好的pre_data放入crf对应文件夹
\cp -r $pre_path "$crf_path"/example/exp/pre_data

# 执行crf识别程序 一个执行程序对应一个需求集
#sh "$crf_path"/example/exp/exec_uav.sh
#sh "$crf_path"/example/exp/exec_gannt.sh
#sh "$crf_path"/example/exp/exec_pure.sh
#sh "$crf_path"/example/exp/exec_worldvista.sh

#sh "$crf_path"/example/exp/exec_external_uav.sh
#sh "$crf_path"/example/exp/exec_external_pure.sh
sh "$crf_path"/example/exp/exec_external_gannt.sh
#sh "$crf_path"/example/exp/exec_external_worldvista.sh


# 将识别后的数据放回structure文件夹
\cp -r "$crf_path"/example/exp/gannt/ "$structure_path"/structure

