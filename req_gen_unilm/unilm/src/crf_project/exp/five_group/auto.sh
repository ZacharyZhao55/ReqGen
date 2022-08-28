#!/bin/bash
root_path=/Users/tuanz_lu/PycharmProjects/Pytorch-learning/exp
crf_path=/Users/tuanz_lu/Desktop/graduation-project/CRF++-0.58
pre_path="$root_path"/pre/
dataset_path="$root_path"/dataset/

python "$root_path"/five_group/operation_pre_stanza.py
python "$root_path"/five_group/agent_pre.py
python "$root_path"/five_group/input_pre.py
python "$root_path"/five_group/output_pre.py
python "$root_path"/five_group/restriction_pre.py
python "$root_path"/five_group/event_pre_stanza.py

# 将特征表整合为五折交叉验证格式
python "$root_path"/formData.py

\cp -r "$pre_path" "$crf_path"/example/exp/pre_data
\cp -r "$dataset_path" "$crf_path"/example/exp/pre_data

sh "$crf_path"/example/exp/exec.sh

\cp -r "$crf_path"/example/exp/output/ "$root_path"/output

