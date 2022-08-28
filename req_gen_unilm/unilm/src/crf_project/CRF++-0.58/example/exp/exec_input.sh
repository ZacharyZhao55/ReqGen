#!/bin/sh
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/input_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/first_input.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/input_model_1110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/input_model_1110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/input_pre_E.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/input_1110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/input_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/second_input.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/input_model_2110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/input_model_2110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/input_pre_A.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/input_2110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/input_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/third_input.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/input_model_3110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/input_model_3110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/input_pre_B.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/input_3110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/input_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/four_input.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/input_model_4110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/input_model_4110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/input_pre_C.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/input_4110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/input_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/five_input.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/input_model_5110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/input_model_5110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/input_pre_D.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/input_5110.output



