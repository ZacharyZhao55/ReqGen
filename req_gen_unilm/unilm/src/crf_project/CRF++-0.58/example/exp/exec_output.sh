#!/bin/sh
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/output_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/first_output.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/output_model_1110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/output_model_1110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/output_pre_E.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/output_1110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/output_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/second_output.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/output_model_2110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/output_model_2110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/output_pre_A.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/output_2110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/output_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/third_output.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/output_model_3110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/output_model_3110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/output_pre_B.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/output_3110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/output_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/four_output.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/output_model_4110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/output_model_4110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/output_pre_C.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/output_4110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/output_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/five_output.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/output_model_5110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/output_model_5110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/output_pre_D.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/output_5110.output


