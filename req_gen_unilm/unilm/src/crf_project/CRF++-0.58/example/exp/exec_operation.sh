#!/bin/sh
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/operation_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/first_operation.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/operation_model_1110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/operation_model_1110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/operation_pre_E.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/operation_1110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/operation_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/second_operation.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/operation_model_2110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/operation_model_2110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/operation_pre_A.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/operation_2110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/operation_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/third_operation.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/operation_model_3110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/operation_model_3110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/operation_pre_B.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/operation_3110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/operation_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/four_operation.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/operation_model_4110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/operation_model_4110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/operation_pre_C.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/operation_4110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/operation_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/five_operation.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/operation_model_5110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/operation_model_5110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/operation_pre_D.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/operation_5110.output